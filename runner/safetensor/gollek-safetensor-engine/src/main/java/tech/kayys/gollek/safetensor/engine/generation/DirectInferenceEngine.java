/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * DirectInferenceEngine.java
 * ──────────────────────────
 * Loads SafeTensor model weights directly using the FFM-based loader
 * (zero-copy mmap) and bridges them to LibTorch Tensors via
 * SafetensorWeightBridge — no GGUF conversion required.
 *
 * Model directory layout expected by this engine:
 *   <model-dir>/
 *     config.json                        — model architecture config
 *     tokenizer.json                     — HF tokenizer vocab + merges
 *     model.safetensors                  — single-file weights
 *     -or-
 *     model.safetensors.index.json       — sharded model index
 *     model-00001-of-NNNNN.safetensors   — weight shards
 *     ...
 *
 * Loading sequence
 * ════════════════
 *   1. SafetensorLoaderFacade.open(modelPath)   — mmap all shard files
 *   2. SafetensorWeightBridge.bridge(tensor)     — zero-copy FFM→LibTorch
 *   3. HuggingFaceTokenizer.load(tokenizer.json) — load BPE tokenizer
 *   4. ModelConfigReader.read(config.json)       — parse architecture info
 *   5. Store in modelsByPath / modelsByKey
 */
package tech.kayys.gollek.safetensor.engine.generation;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.inference.libtorch.core.Device;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.loader.SafetensorLoaderFacade;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.gollek.safetensor.loader.SafetensorTensor;
import tech.kayys.gollek.safetensor.loader.SafetensorDType;
import tech.kayys.gollek.safetensor.engine.warmup.SafetensorWeightBridge;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.StreamingDecoder;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.safetensor.models.ModelArchitectureRegistry;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.function.Supplier;
import java.util.concurrent.Executors;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.TimeUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.HashSet;
import java.util.Random;
import java.util.UUID;
import java.util.Map;
import java.util.Objects;
import java.util.Collection;
import java.util.Collections;

/**
 * Direct, zero-copy SafeTensor inference engine.
 *
 * <p>Loads model weights from {@code .safetensors} files via memory-mapped
 * I/O (no GGUF conversion). Compatible with both single-file and sharded
 * HuggingFace Hub model checkpoints.
 *
 * <p>Inject via CDI:
 * <pre>{@code
 * @Inject DirectInferenceEngine engine;
 * String key = engine.loadModel(Path.of("/models/llama3-8b"));
 * LoadedModel model = engine.getLoadedModel(key);
 * }</pre>
 */
@ApplicationScoped
public class DirectInferenceEngine implements SafetensorEngine {

    private static final Logger log = Logger.getLogger(DirectInferenceEngine.class);

    // ── CDI dependencies ─────────────────────────────────────────────────────

    /** FFM-backed loader: handles single-file and multi-shard models. */
    @Inject
    SafetensorLoaderFacade safetensorLoader;

    /** Zero-copy bridge from SafetensorTensor (FFM) to LibTorch TorchTensor. */
    @Inject
    SafetensorWeightBridge bridge;

    /** Quantization support for INT4/INT8/FP8 models. */
    @Inject
    QuantizationEngine quantizationEngine;

    /** Jackson mapper for reading config.json. */
    @Inject
    ObjectMapper objectMapper;

    @Inject
    DirectForwardPass forwardPass;
    @Inject
    TokenSampler tokenSampler;
    @Inject
    ModelArchitectureRegistry archRegistry;

    // ── Model registry ────────────────────────────────────────────────────────

    private final Map<Path, LoadedModel> modelsByPath = new ConcurrentHashMap<>();
    private final Map<String, LoadedModel> modelsByKey = new ConcurrentHashMap<>();
    private final KVCacheManager kvCacheManager = new KVCacheManager();

    // ─────────────────────────────────────────────────────────────────────────
    // LoadedModel
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * A loaded model held in memory, backed by zero-copy mmap'd weight tensors.
     */
    public static class LoadedModel implements SafetensorEngine.LoadedModel {
        private final Path path;
        private final Tokenizer tokenizer;
        private final String key;
        private final boolean quantized;
        private final QuantizationEngine.QuantStrategy quantStrategy;
        private final ModelConfig config;

        private final Map<String, TorchTensor> weights;
        private final Arena weightArena;

        public LoadedModel(Path path, Map<String, TorchTensor> weights,
                Tokenizer tokenizer, String key,
                boolean quantized, QuantizationEngine.QuantStrategy quantStrategy,
                ModelConfig config, Arena weightArena) {
            this.path = path;
            this.weights = weights;
            this.tokenizer = tokenizer;
            this.key = key;
            this.quantized = quantized;
            this.quantStrategy = quantStrategy;
            this.config = config != null ? config : new ModelConfig();
            this.weightArena = weightArena;
        }

        @Override public Path path() { return path; }
        @Override public Map<String, TorchTensor> weights() { return weights; }
        @Override public Tokenizer tokenizer() { return tokenizer; }
        @Override public String key() { return key; }
        @Override public boolean isQuantized() { return quantized; }

        /** Architecture and dimension config parsed from {@code config.json}. */
        @Override
        public ModelConfig config() { return config; }

        /** The model architecture string (e.g. "LlamaForCausalLM"). */
        public String arch() { return config.primaryArchitecture(); }

        public QuantizationEngine.QuantStrategy getQuantStrategy() { return quantStrategy; }
    }

    // Removed local ModelConfig record — now using tech.kayys.gollek.spi.model.ModelConfig

    // ─────────────────────────────────────────────────────────────────────────
    // Public API — InferenceEngine
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Load a SafeTensor model from {@code modelPath} without quantization.
     *
     * @param modelPath path to the model file or directory
     * @return a stable key identifying this model in the registry
     */
    @Override
    public String loadModel(Path modelPath) {
        return loadModel(modelPath, null, QuantizationEngine.QuantStrategy.NONE);
    }

    /**
     * Load a SafeTensor model with optional LoRA adapter and quantization.
     *
     * @param modelPath     path to the model file or model directory
     * @param adapterPath   optional LoRA adapter path (may be null)
     * @param quantStrategy quantization strategy to apply on weight load
     * @return stable model key
     */
    public String loadModel(Path modelPath, Path adapterPath,
            QuantizationEngine.QuantStrategy quantStrategy) {

        Objects.requireNonNull(modelPath, "modelPath must not be null");
        Path resolved = modelPath.toAbsolutePath().normalize();

        // ── Already loaded? ────────────────────────────────────────────────
        if (modelsByPath.containsKey(resolved)) {
            log.infof("DirectInferenceEngine: model already loaded [%s]", resolved.getFileName());
            return modelsByPath.get(resolved).key();
        }

        log.infof("DirectInferenceEngine: loading model [%s] strategy=%s", resolved.getFileName(), quantStrategy);

        // ── Load weights via FFM mmap ──────────────────────────────────────
        Arena weightArena = Arena.ofShared();
        Map<String, TorchTensor> weights;
        try {
            weights = (quantStrategy == QuantizationEngine.QuantStrategy.NONE)
                    ? loadWeights(resolved, weightArena)
                    : quantizationEngine.loadQuantizedModel(resolved, quantStrategy);
        } catch (Exception e) {
            weightArena.close();
            throw e;
        }

        // ── Load tokenizer ─────────────────────────────────────────────────
        Tokenizer tokenizer;
        try {
            tokenizer = loadTokenizer(resolved);
        } catch (Exception e) {
            log.warnf("Failed to load tokenizer for [%s], using dummy", resolved);
            // We should really have a dummy tokenizer implementation
            throw new RuntimeException("Tokenizer loading failed", e);
        }

        // ── Parse config.json ──────────────────────────────────────────────
        ModelConfig config = loadConfig(resolved);

        // ── Register ───────────────────────────────────────────────────────
        String key = resolved.getFileName().toString();
        LoadedModel model = new LoadedModel(resolved, weights, tokenizer, key,
                quantStrategy != QuantizationEngine.QuantStrategy.NONE, quantStrategy, config, weightArena);

        modelsByPath.put(resolved, model);
        modelsByKey.put(key, model);

        log.infof("DirectInferenceEngine: loaded [%s] — %d weights, arch=%s, quantized=%s",
                key, weights.size(), config.modelType(), quantStrategy);
        return key;
    }

    /**
     * Load an already-quantized SafeTensor model (INT4/INT8/FP8).
     */
    public String loadQuantizedModel(Path quantizedModelPath,
            QuantizationEngine.QuantStrategy quantStrategy) {
        return loadModel(quantizedModelPath, null, quantStrategy);
    }

    /**
     * Generate text from a prompt.
     *
     * @param prompt    text prompt
     * @param modelPath loaded model path
     * @param cfg       generation config
     * @return generated text response
     */
    public Uni<InferenceResponse> generate(String prompt, Path modelPath, GenerationConfig cfg) {
                log.debugf("DirectInferenceEngine: Model loaded with %d weights", model.weights().size());
                
                Tokenizer tokenizer = model.tokenizer();
                ModelConfig config = model.config();
                ModelArchitecture arch = archRegistry.resolve(config);

                if (!model.weights().isEmpty()) {
                    try {
                        TorchTensor firstWeight = model.weights().values().iterator().next();
                        tech.kayys.gollek.inference.libtorch.core.Device device = firstWeight.getDevice();
                        log.infof("DirectInferenceEngine: model=%s, device=%s, weights=%d",
                            model.key(), device, model.weights().size());
                    } catch (Exception e) {
                        log.warnf("Could not log device info: %s", e.getMessage());
                    }
                }

                long[] inputIds = tokenizer.encode(prompt, EncodeOptions.defaultOptions());
                inputLen = inputIds.length;
                
                try (KVCacheManager.KVCacheSession session = kvCacheManager.createSession(cfg.maxKvCacheTokens())) {
                    session.allocate(config, model.weights(), arch);
                    
                    float[] logits = forwardPass.prefill(inputIds, model.weights(), config, arch, session);
                    applyLogitSoftcap(logits, config);

                    int[] freq = new int[config.vocabSize()];
                    Random rng = new Random();
                    int next = tokenSampler.sample(logits, cfg, freq, rng);

                    List<Long> generatedIds = new ArrayList<>();
                    Set<Integer> stops = new HashSet<>();
                    stops.add(tokenizer.eosTokenId());
                    stops.addAll(cfg.stopTokenIds());

                    StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                    for (int step = 0; step < cfg.maxNewTokens(); step++) {
                        String delta = decoder.decodeNext((long) next);
                        out.append(delta);
                        
                        String fullGeneratedText = decoder.currentText();

                        // Check for ID-based stops
                        if (stops.contains(next)) break;
                        
                        // Check for String-based stops (e.g., <|im_end|>)
                        if (!cfg.stopStrings().isEmpty()) {
                            boolean shouldStop = false;
                            for (String s : cfg.stopStrings()) {
                                if (fullGeneratedText.endsWith(s)) {
                                    shouldStop = true;
                                    break;
                                }
                            }
                            if (shouldStop) break;
                        }

                        if (next >= 0 && next < freq.length) freq[next]++;
                        
                        logits = forwardPass.decode(next, inputIds.length + step, model.weights(), config, arch, session);
                        applyLogitSoftcap(logits, config);
                        next = tokenSampler.sample(logits, cfg, freq, rng);
                    }
                }

                return InferenceResponse.builder()
                        .requestId(UUID.randomUUID().toString())
                        .content(out.toString())
                        .model(modelPath.getFileName().toString())
                        .inputTokens(inputLen)
                        .outputTokens(out.length() / 4) // rough estimate
                        .durationMs(java.time.Duration.between(t0, Instant.now()).toMillis())
                        .finishReason(InferenceResponse.FinishReason.STOP)
                        .metadata("backend", "direct-safetensor")
                        .build();
            } catch (Exception e) {
                log.error("Generation failed", e);
                throw new RuntimeException("Direct generation failed: " + e.getMessage(), e);
            }
        }).runSubscriptionOn(Executors.newVirtualThreadPerTaskExecutor());
    }

    /**
     * Generate text as a stream of responses.
     */
    public Multi<InferenceResponse> generateStream(String prompt, Path modelPath, GenerationConfig cfg) {
                if (!model.weights().isEmpty()) {
                    try {
                        TorchTensor firstWeight = model.weights().values().iterator().next();
                        tech.kayys.gollek.inference.libtorch.core.Device device = firstWeight.getDevice();
                        log.infof("generateStream: model=%s, device=%s, weights=%d",
                            model.key(), device, model.weights().size());
                    } catch (Exception e) {
                        log.warnf("Could not log device info: %s", e.getMessage());
                    }
                }
                
                Tokenizer tokenizer = model.tokenizer();
                ModelConfig config = model.config();
                ModelArchitecture arch = archRegistry.resolve(config);

                long[] inputIds = tokenizer.encode(prompt, EncodeOptions.defaultOptions());
                inputLen = inputIds.length;
                
                try (KVCacheManager.KVCacheSession session = kvCacheManager.createSession(cfg.maxKvCacheTokens())) {
                    session.allocate(config, model.weights(), arch);
                    
                    float[] logits = forwardPass.prefill(inputIds, model.weights(), config, arch, session);
                    applyLogitSoftcap(logits, config);

                    int[] freq = new int[config.vocabSize()];
                    Random rng = new Random();
                    int next = tokenSampler.sample(logits, cfg, freq, rng);

                    Set<Integer> stops = new HashSet<>();
                    stops.add(tokenizer.eosTokenId());
                    stops.addAll(cfg.stopTokenIds());
                    
                    StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                    for (int step = 0; step < cfg.maxNewTokens(); step++) {
                        String delta = decoder.decodeNext((long) next);
                        String fullGeneratedText = decoder.currentText();
                        
                        if (!delta.isEmpty()) {
                            publisher.submit(InferenceResponse.builder()
                                    .requestId(requestId)
                                    .content(delta)
                                    .model(modelPath.getFileName().toString())
                                    .inputTokens(inputLen)
                                    .metadata("backend", "direct-safetensor")
                                    .build());
                        }

                        // Check ID-based stops
                        if (stops.contains(next)) break;

                        // Check String-based stops
                        if (!cfg.stopStrings().isEmpty()) {
                            boolean shouldStop = false;
                            for (String s : cfg.stopStrings()) {
                                if (fullGeneratedText.endsWith(s)) {
                                    shouldStop = true;
                                    break;
                                }
                            }
                            if (shouldStop) break;
                        }

                        if (next >= 0 && next < freq.length) freq[next]++;
                        
                        logits = forwardPass.decode(next, inputIds.length + step, model.weights(), config, arch, session);
                        applyLogitSoftcap(logits, config);
                        next = tokenSampler.sample(logits, cfg, freq, rng);
                    }
                }

                publisher.submit(InferenceResponse.builder()
                        .requestId(requestId)
                        .content("")
                        .model(modelPath.getFileName().toString())
                        .durationMs(java.time.Duration.between(t0, Instant.now()).toMillis())
                        .finishReason(InferenceResponse.FinishReason.STOP)
                        .metadata("backend", "direct-safetensor")
                        .build());

            } catch (Exception e) {
                log.error("Stream generation failed", e);
                publisher.closeExceptionally(e);
            } finally {
                publisher.close();
            }
        });

        return Multi.createFrom().publisher(publisher);
    }

    @Override
    public void unloadModel(Path modelPath) {
        Path resolved = modelPath.toAbsolutePath().normalize();
        LoadedModel model = modelsByPath.remove(resolved);
        if (model != null) {
            modelsByKey.remove(model.key());
            model.weights().values().forEach(t -> {
                try { t.close(); } catch (Exception ignored) { }
            });
            if (model.weightArena != null) {
                try { model.weightArena.close(); } catch (Exception ignored) { }
            }
            log.infof("DirectInferenceEngine: unloaded [%s]", resolved.getFileName());
        }
    }

    @Override
    public LoadedModel getLoadedModel(Path modelPath) {
        return modelsByPath.get(modelPath.toAbsolutePath().normalize());
    }

    @Override
    public LoadedModel getLoadedModel(String key) {
        return modelsByKey.get(key);
    }

    /** Whether this model path is currently loaded. */
    public boolean isLoaded(Path modelPath) {
        return modelsByPath.containsKey(modelPath.toAbsolutePath().normalize());
    }

    /** All currently loaded models. */
    public Collection<LoadedModel> listLoadedModels() {
        return Collections.unmodifiableCollection(modelsByPath.values());
    }

    /** Access the quantization engine for caller-driven quantization. */
    public QuantizationEngine getQuantizationEngine() { return quantizationEngine; }

    /** KV-cache manager for generation sessions. */
    public KVCacheManager getKVCacheManager() { return kvCacheManager; }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal — weight loading via SafetensorLoaderFacade + WeightBridge
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Open the model via the FFM loader facade, iterate all tensor names, and
     * bridge each one to a LibTorch TorchTensor (zero-copy).
     *
     * <p>The weight session is a try-with-resources scope; after the loop the
     * mmap'd file regions remain accessible through the bridged LibTorch tensors
     * because the WeightBridge creates a new shared Arena for each tensor.
     */
    private Map<String, TorchTensor> loadWeights(Path modelPath, Arena weightArena) {
        log.debugf("DirectInferenceEngine: opening weights from [%s]", modelPath);
        
        // Detect if Metal device is available and should be used
        Device targetDevice = detectMetalDevice();
        log.infof("DirectInferenceEngine: loading weights to device: %s", targetDevice);
        
        Map<String, TorchTensor> weights = new ConcurrentHashMap<>();
        try (SafetensorShardSession session = safetensorLoader.open(modelPath)) {
            Set<String> names = session.tensorNames();
            log.debugf("DirectInferenceEngine: bridging %d tensors to LibTorch on device %s", names.size(), targetDevice);
            
            int movedCount = 0;
            int failedCount = 0;
            
            for (String name : names) {
                SafetensorTensor st = session.tensor(name);
                // Standardize on Float32 for inference stability
                TorchTensor t = bridge.bridge(st);
                
                // Move tensor to target device (Metal if available, otherwise CPU)
                if (!targetDevice.isCpu()) {
                    try {
                        TorchTensor moved = t.to(targetDevice);
                        t.close(); // Close the CPU-bound bridged tensor
                        t = moved;
                        log.debugf("Moved tensor %s to device %s", name, targetDevice);
                        movedCount++;
                    } catch (Exception e) {
                        failedCount++;
                        log.warnf("Failed to move tensor %s to device %s, keeping on CPU: %s", name, targetDevice, e.getMessage());
                        // Keep tensor 't' on CPU if device move fails
                    }
                }
                weights.put(name, t);
            }
            log.debugf("Bridge complete: %d moved, %d failed", movedCount, failedCount);
        } catch (tech.kayys.gollek.spi.exception.ProviderException e) {
            log.errorf("Failed to load weights from %s: %s", modelPath, e.getMessage());
            throw new IllegalStateException("Failed to load weights from " + modelPath + ": " + e.getMessage(), e);
        }
        applyWeightAliases(weights);
        log.debugf("DirectInferenceEngine: bridged %d tensors to device %s", weights.size(), targetDevice);
        return weights;
    }
    
    /**
     * Detect if Metal device is available and should be used for inference.
     * Checks for Apple Silicon with Metal enabled via environment variables.
     */
    private Device detectMetalDevice() {
        // Check if running on Apple Silicon
        String osName = System.getProperty("os.name", "").toLowerCase();
        String osArch = System.getProperty("os.arch", "").toLowerCase();
        log.debugf("Platform detection: OS=%s, Arch=%s", osName, osArch);
        
        boolean isAppleSilicon = osName.contains("mac") && (osArch.contains("aarch64") || osArch.contains("arm64"));
        
        if (!isAppleSilicon) {
            log.infof("Not Apple Silicon, using CPU device");
            return Device.CPU;
        }
        
        // Check if Metal is disabled via environment variable
        String metalDisabled = System.getenv("GOLLEK_METAL_ENABLED");
        if (metalDisabled != null && metalDisabled.equalsIgnoreCase("false")) {
            log.info("DirectInferenceEngine: Metal disabled via GOLLEK_METAL_ENABLED=false");
            return Device.CPU;
        }
        
        // Use MPS (Metal Performance Shaders) device
        log.info("DirectInferenceEngine: Using MPS (Metal Performance Shaders) for inference");
        return Device.MPS;
    }

    private void applyWeightAliases(Map<String, TorchTensor> weights) {
        // Some multimodal checkpoints nest the text model under a prefix.
        List<Alias> aliases = List.of(
                new Alias("text_model.", "model."),
                new Alias("model.text_model.", "model."),
                new Alias("language_model.model.", "model."),
                new Alias("model.language_model.", "model."),
                new Alias("text_model.model.", "model."),
                new Alias("model.text_model.model.", "model.")
        );
        for (Alias alias : aliases) {
            boolean hasPrefix = weights.keySet().stream().anyMatch(k -> k.startsWith(alias.from()));
            if (!hasPrefix) {
                continue;
            }
            for (Map.Entry<String, TorchTensor> entry : weights.entrySet()) {
                String key = entry.getKey();
                if (!key.startsWith(alias.from())) {
                    continue;
                }
                String rewritten = alias.to() + key.substring(alias.from().length());
                weights.putIfAbsent(rewritten, entry.getValue());
            }
            log.debugf("DirectInferenceEngine: applied weight alias %s -> %s", alias.from(), alias.to());
        }
    }

    private record Alias(String from, String to) {}

    // ─────────────────────────────────────────────────────────────────────────
    // Internal — tokenizer loading
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Discover and load {@code tokenizer.json} from the model directory.
     *
     * <p>Search order:
     * <ol>
     *   <li>{@code modelPath/tokenizer.json} (directory layout)</li>
     *   <li>{@code parent(modelPath)/tokenizer.json} (single-file layout)</li>
     * </ol>
     * Falls back to a character-level tokenizer only if neither file is found.
     */
    private Tokenizer loadTokenizer(Path modelPath) {
        try {
            return TokenizerFactory.load(modelPath, null);
        } catch (IOException e) {
            Path modelDir = Files.isRegularFile(modelPath) ? modelPath.getParent() : modelPath;
            
            // Try tokenizer/ subdirectory (common in diffusers pipelines)
            if (modelDir != null && Files.isDirectory(modelDir.resolve("tokenizer"))) {
                try {
                    return TokenizerFactory.load(modelDir.resolve("tokenizer"), null);
                } catch (IOException ignored) {
                    // Fall back to original error reporting
                }
            }

            // Check root and subdirectories
            boolean rootJson = modelDir != null && Files.exists(modelDir.resolve("tokenizer.json"));
            boolean subJson = modelDir != null && Files.exists(modelDir.resolve("tokenizer").resolve("tokenizer.json"));
            boolean rootVocab = modelDir != null && Files.exists(modelDir.resolve("vocab.json"));
            boolean subVocab = modelDir != null && Files.exists(modelDir.resolve("tokenizer").resolve("vocab.json"));
            
            log.errorf(e, "Tokenizer loading failed for [%s]. Files found: root/tokenizer.json=%s, sub/tokenizer.json=%s, root/vocab.json=%s, sub/vocab.json=%s",
                    modelPath, rootJson, subJson, rootVocab, subVocab);
            throw new RuntimeException("Tokenizer loading failed. Ensure the model directory contains a supported tokenizer structure. Error: " + e.getMessage(), e);
        }
    }

    /**
     * Parse model architecture metadata from {@code config.json}.
     */
    private ModelConfig loadConfig(Path modelPath) {
        try {
            Path configDir = modelPath;
            if (Files.isRegularFile(modelPath)) {
                configDir = modelPath.getParent();
            }
            if (configDir != null) {
                // If it's a pipeline directory, look into subcomponents
                if (Files.exists(configDir.resolve("model_index.json"))) {
                    // Try unet for SD or text_encoder for others
                    if (Files.isDirectory(configDir.resolve("unet")) && Files.exists(configDir.resolve("unet").resolve("config.json"))) {
                        return ModelConfig.fromDirectory(configDir.resolve("unet"), objectMapper);
                    } else if (Files.isDirectory(configDir.resolve("text_encoder")) && Files.exists(configDir.resolve("text_encoder").resolve("config.json"))) {
                        return ModelConfig.fromDirectory(configDir.resolve("text_encoder"), objectMapper);
                    }
                }
                return ModelConfig.fromDirectory(configDir, objectMapper);
            }
            return new ModelConfig();
        } catch (IOException e) {
            log.warnf(e, "DirectInferenceEngine: failed to read config.json near [%s]", modelPath);
            return new ModelConfig();
        }
    }

    private static void applyLogitSoftcap(float[] logits, ModelConfig config) {
        if (logits == null) return;
        Double cap = config.finalLogitSoftcapping();
        if (cap == null || cap <= 0) return;
        float c = cap.floatValue();
        for (int i = 0; i < logits.length; i++) {
            float x = logits[i];
            logits[i] = (float) (c * Math.tanh(x / c));
        }
    }
}
