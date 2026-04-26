/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * DirectInferenceEngine.java
 * ──────────────────────────
 * Loads SafeTensor model weights using FFM mmap and bridges them to
 * AccelTensor — pure Java + Apple Accelerate. No LibTorch dependency.
 */
package tech.kayys.gollek.safetensor.engine.generation;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.loader.SafetensorLoaderFacade;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.gollek.safetensor.quantization.bridge.AccelWeightBridge;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.StreamingDecoder;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.models.core.ModelArchitectureRegistry;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.function.Supplier;
import java.util.concurrent.Executors;
import java.util.concurrent.ConcurrentHashMap;
import java.util.HashMap;
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
 * Direct SafeTensor inference engine using AccelTensor + Apple Accelerate.
 * No LibTorch dependency.
 */
@ApplicationScoped
public class DirectInferenceEngine implements SafetensorEngine {

    private static final Logger log = Logger.getLogger(DirectInferenceEngine.class);

    @Inject
    SafetensorLoaderFacade safetensorLoader;

    @Inject
    AccelWeightBridge bridge;

    @Inject
    QuantizationEngine quantizationEngine;

    @Inject
    ObjectMapper objectMapper;

    @Inject
    DirectForwardPass forwardPass;
    @Inject
    TokenSampler tokenSampler;
    @Inject
    ModelArchitectureRegistry archRegistry;

    @Inject
    KVCacheManager kvCacheManager;

    private final Map<Path, LoadedModel> modelsByPath = new ConcurrentHashMap<>();
    private final Map<String, LoadedModel> modelsByKey = new ConcurrentHashMap<>();
    private final Map<Path, Object> modelLocks = new ConcurrentHashMap<>();

    // ─────────────────────────────────────────────────────────────────────────
    // LoadedModel
    // ─────────────────────────────────────────────────────────────────────────

    public static class LoadedModel implements SafetensorEngine.LoadedModel {
        private final Path path;
        private final Tokenizer tokenizer;
        private final String key;
        private final boolean quantized;
        private final QuantizationEngine.QuantStrategy quantStrategy;
        private final ModelConfig config;
        private final Map<String, AccelTensor> weights;
        private final Arena weightArena;

        public LoadedModel(Path path, Map<String, AccelTensor> weights,
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

        @Override
        public Path path() {
            return path;
        }

        @Override
        public Map<String, AccelTensor> weights() {
            return weights;
        }

        @Override
        public Tokenizer tokenizer() {
            return tokenizer;
        }

        @Override
        public String key() {
            return key;
        }

        @Override
        public boolean isQuantized() {
            return quantized;
        }

        @Override
        public ModelConfig config() {
            return config;
        }

        public String arch() {
            return config.primaryArchitecture();
        }

        public QuantizationEngine.QuantStrategy getQuantStrategy() {
            return quantStrategy;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public String loadModel(Path modelPath) {
        return loadModel(modelPath, null, QuantizationEngine.QuantStrategy.NONE);
    }

    public String loadModel(Path modelPath, Path adapterPath,
            QuantizationEngine.QuantStrategy quantStrategy) {

        Objects.requireNonNull(modelPath, "modelPath must not be null");
        Path resolved = modelPath.toAbsolutePath().normalize();

        if (modelsByPath.containsKey(resolved)) {
            log.infof("DirectInferenceEngine: model already loaded [%s]", resolved.getFileName());
            return modelsByPath.get(resolved).key();
        }

        synchronized (modelLocks.computeIfAbsent(resolved, k -> new Object())) {
            // Double-check inside lock
            if (modelsByPath.containsKey(resolved)) {
                return modelsByPath.get(resolved).key();
            }

            log.infof("DirectInferenceEngine: loading model [%s]", resolved.getFileName());

            Arena weightArena = Arena.ofAuto();
            Map<String, AccelTensor> weights = loadWeights(resolved);

            Tokenizer tokenizer;
            try {
                tokenizer = loadTokenizer(resolved);
            } catch (Exception e) {
                log.warnf("Failed to load tokenizer for [%s]", resolved);
                throw new RuntimeException("Tokenizer loading failed", e);
            }

            ModelConfig config = loadConfig(resolved);

            String key = resolved.getFileName().toString();
            LoadedModel model = new LoadedModel(resolved, weights, tokenizer, key,
                    quantStrategy != QuantizationEngine.QuantStrategy.NONE, quantStrategy, config, weightArena);

            modelsByPath.put(resolved, model);
            modelsByKey.put(key, model);

            log.infof("DirectInferenceEngine: loaded [%s] — %d weights, arch=%s (Accelerate backend)",
                    key, weights.size(), config.modelType());
            return key;
        }
    }

    public Uni<InferenceResponse> generate(String prompt, Path modelPath, GenerationConfig cfg) {
        return Uni.createFrom().item((Supplier<InferenceResponse>) () -> {
            Instant t0 = Instant.now();
            StringBuilder out = new StringBuilder();
            int inputLen = 0;

            try {
                LoadedModel model = (LoadedModel) getLoadedModel(modelPath);
                if (model == null) {
                    loadModel(modelPath);
                    model = (LoadedModel) getLoadedModel(modelPath);
                }

                if (model == null)
                    throw new RuntimeException("Model failed to load");

                Tokenizer tokenizer = model.tokenizer();
                ModelConfig config = model.config();
                ModelArchitecture arch = archRegistry.resolve(config);

                long[] inputIds = tokenizer.encode(prompt, EncodeOptions.defaultOptions());
                inputLen = inputIds.length;

                try (KVCacheManager.KVCacheSession session = kvCacheManager.createSession(cfg.maxKvCacheTokens())) {
                    session.allocate(config);

                    float[] logits = forwardPass.prefill(inputIds, model.weights(), config, arch, session);
                    applyLogitSoftcap(logits, config);

                    int[] freq = new int[config.vocabSize()];
                    Random rng = new Random();
                    int next = tokenSampler.sample(logits, cfg, freq, rng);

                    Set<Integer> stops = new HashSet<>();
                    for (int id : tokenizer.allStopTokenIds())
                        stops.add(id);
                    stops.addAll(cfg.stopTokenIds());

                    StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                    for (int step = 0; step < cfg.maxNewTokens(); step++) {
                        if (stops.contains(next))
                            break;

                        String delta = decoder.decodeNext((long) next);

                        out.append(delta);
                        String fullGeneratedText = decoder.currentText();
                        if (!cfg.stopStrings().isEmpty()) {
                            boolean shouldStop = false;
                            for (String s : cfg.stopStrings()) {
                                if (fullGeneratedText.endsWith(s)) {
                                    shouldStop = true;
                                    break;
                                }
                            }
                            if (shouldStop)
                                break;
                        }

                        if (next >= 0 && next < freq.length)
                            freq[next]++;
                        logits = forwardPass.decode(next, inputIds.length + step, model.weights(), config, arch,
                                session);
                        applyLogitSoftcap(logits, config);
                        next = tokenSampler.sample(logits, cfg, freq, rng);
                    }
                }

                return InferenceResponse.builder()
                        .requestId(UUID.randomUUID().toString())
                        .content(out.toString())
                        .model(modelPath.getFileName().toString())
                        .inputTokens(inputLen)
                        .outputTokens(out.length() / 4)
                        .durationMs(java.time.Duration.between(t0, Instant.now()).toMillis())
                        .finishReason(InferenceResponse.FinishReason.STOP)
                        .metadata("backend", "accelerate-safetensor")
                        .build();
            } catch (Exception e) {
                log.error("Generation failed", e);
                throw new RuntimeException("Direct generation failed: " + e.getMessage(), e);
            }
        }).runSubscriptionOn(Executors.newVirtualThreadPerTaskExecutor());
    }

    public Multi<InferenceResponse> generateStream(String prompt, Path modelPath, GenerationConfig cfg) {
        return Multi.createFrom().emitter(emitter -> {
            Executors.newVirtualThreadPerTaskExecutor().submit(() -> {
                Instant t0 = Instant.now();
                String requestId = UUID.randomUUID().toString();
                int inputLen = 0;

                try {
                    LoadedModel model = (LoadedModel) getLoadedModel(modelPath);
                    if (model == null) {
                        loadModel(modelPath);
                        model = (LoadedModel) getLoadedModel(modelPath);
                    }

                    if (model == null)
                        throw new RuntimeException("Model failed to load");

                    Tokenizer tokenizer = model.tokenizer();
                    ModelConfig config = model.config();
                    ModelArchitecture arch = archRegistry.resolve(config);

                    // Runtime metadata logging for stability verification

                    long[] inputIds = tokenizer.encode(prompt, EncodeOptions.defaultOptions());
                    inputLen = inputIds.length;

                    try (KVCacheManager.KVCacheSession session = kvCacheManager.createSession(cfg.maxKvCacheTokens())) {
                        session.allocate(config);

                        float[] logits = forwardPass.prefill(inputIds, model.weights(), config, arch, session);
                        applyLogitSoftcap(logits, config);

                        int[] freq = new int[config.vocabSize()];

                        Random rng = new Random();
                        int next = tokenSampler.sample(logits, cfg, freq, rng);

                        Set<Integer> stops = new HashSet<>();
                        for (int id : tokenizer.allStopTokenIds())
                            stops.add(id);
                        stops.addAll(cfg.stopTokenIds());

                        StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                        for (int step = 0; step < cfg.maxNewTokens(); step++) {

                            if (stops.contains(next) || emitter.isCancelled()) {
                                break;
                            }

                            String delta = decoder.decodeNext((long) next);
                            if (delta == null)
                                delta = "";

                            String fullGeneratedText = decoder.currentText();

                            if (!delta.isEmpty()) {
                                emitter.emit(InferenceResponse.builder()
                                        .requestId(requestId)
                                        .content(delta)
                                        .model(modelPath.getFileName().toString())
                                        .inputTokens(inputLen)
                                        .metadata("backend", "accelerate-safetensor")
                                        .build());
                            }

                            if (!cfg.stopStrings().isEmpty()) {
                                boolean shouldStop = false;
                                for (String s : cfg.stopStrings()) {
                                    if (fullGeneratedText.endsWith(s)) {
                                        shouldStop = true;
                                        break;
                                    }
                                }
                                if (shouldStop)
                                    break;
                            }

                            if (next >= 0 && next < freq.length)
                                freq[next]++;
                            int decodeStartPos = inputIds.length + step;
                            logits = forwardPass.decode(next, decodeStartPos, model.weights(), config, arch,
                                    session);
                            applyLogitSoftcap(logits, config);

                            // Step-level diagnostics for first 20 tokens
                            if (step < 20) {
                                int topId = 0;
                                float topVal = logits[0];
                                int top2Id = 0;
                                float top2Val = Float.NEGATIVE_INFINITY;
                                int top3Id = 0;
                                float top3Val = Float.NEGATIVE_INFINITY;
                                for (int di = 1; di < logits.length; di++) {
                                    if (logits[di] > topVal) {
                                        top3Id = top2Id;
                                        top3Val = top2Val;
                                        top2Id = topId;
                                        top2Val = topVal;
                                        topVal = logits[di];
                                        topId = di;
                                    } else if (logits[di] > top2Val) {
                                        top3Id = top2Id;
                                        top3Val = top2Val;
                                        top2Val = logits[di];
                                        top2Id = di;
                                    } else if (logits[di] > top3Val) {
                                        top3Val = logits[di];
                                        top3Id = di;
                                    }
                                }
                                System.err.printf(
                                        "[DIAG] step=%d tok=%d startPos=%d kvPos=%d top3=[%d(%.2f) %d(%.2f) %d(%.2f)]%n",
                                        step, next, decodeStartPos, session.currentPos(), topId, topVal, top2Id,
                                        top2Val, top3Id, top3Val);
                            }

                            next = tokenSampler.sample(logits, cfg, freq, rng);
                        }
                    }

                    emitter.emit(InferenceResponse.builder()
                            .requestId(requestId)
                            .content("")
                            .model(modelPath.getFileName().toString())
                            .durationMs(java.time.Duration.between(t0, Instant.now()).toMillis())
                            .finishReason(InferenceResponse.FinishReason.STOP)
                            .inputTokens(inputLen)
                            .metadata("backend", "accelerate-safetensor")
                            .build());

                } catch (Throwable t) {
                    log.error("Generation failed", t);
                    emitter.fail(t);
                } finally {
                    emitter.complete();
                }
            });
        });
    }

    @Override
    public void unloadModel(Path modelPath) {
        Path resolved = modelPath.toAbsolutePath().normalize();
        LoadedModel model = modelsByPath.remove(resolved);
        if (model != null) {
            modelsByKey.remove(model.key());
            model.weights().values().forEach(t -> {
                try {
                    t.close();
                } catch (Exception ignored) {
                }
            });
            if (model.weightArena != null) {
                try {
                    model.weightArena.close();
                } catch (Exception ignored) {
                }
            }
            log.infof("DirectInferenceEngine: unloaded [%s]", resolved.getFileName());
        }
    }

    @Override
    public SafetensorEngine.LoadedModel getLoadedModel(Path modelPath) {
        return modelsByPath.get(modelPath.toAbsolutePath().normalize());
    }

    @Override
    public SafetensorEngine.LoadedModel getLoadedModel(String key) {
        return modelsByKey.get(key);
    }

    public boolean isLoaded(Path modelPath) {
        return modelsByPath.containsKey(modelPath.toAbsolutePath().normalize());
    }

    public Collection<LoadedModel> listLoadedModels() {
        return Collections.unmodifiableCollection(modelsByPath.values());
    }

    public QuantizationEngine getQuantizationEngine() {
        return quantizationEngine;
    }

    public KVCacheManager getKVCacheManager() {
        return kvCacheManager;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Weight loading — pure AccelTensor, no LibTorch
    // ─────────────────────────────────────────────────────────────────────────

    private Map<String, AccelTensor> loadWeights(Path modelPath) {
        log.debugf("DirectInferenceEngine: opening weights from [%s]", modelPath);

        Map<String, AccelTensor> weights;
        try (SafetensorShardSession session = safetensorLoader.open(modelPath)) {
            weights = bridge.bridgeAll(session);
        } catch (Exception e) {
            log.errorf("Failed to load weights from %s: %s", modelPath, e.getMessage());
            throw new IllegalStateException("Failed to load weights: " + e.getMessage(), e);
        }

        applyWeightAliases(weights);
        log.infof("DirectInferenceEngine: bridged %d tensors (Accelerate/FFM backend)", weights.size());
        return weights;
    }

    private void applyWeightAliases(Map<String, AccelTensor> weights) {
        List<Alias> aliases = List.of(
                new Alias("text_model.", "model."),
                new Alias("model.text_model.", "model."),
                new Alias("language_model.model.", "model."),
                new Alias("model.language_model.", "model."),
                new Alias("text_model.model.", "model."),
                new Alias("model.text_model.model.", "model."));

        Map<String, AccelTensor> updates = new HashMap<>();
        // Defensive copy of entries to prevent CME if another thread somehow
        // modifications the map
        List<Map.Entry<String, AccelTensor>> entries = new ArrayList<>(weights.entrySet());

        for (Alias alias : aliases) {
            for (Map.Entry<String, AccelTensor> entry : entries) {
                String key = entry.getKey();
                if (key.startsWith(alias.from())) {
                    String rewritten = alias.to() + key.substring(alias.from().length());
                    if (!weights.containsKey(rewritten)) {
                        updates.putIfAbsent(rewritten, entry.getValue());
                    }
                }
            }
        }
        weights.putAll(updates);
    }

    private record Alias(String from, String to) {
    }

    private Tokenizer loadTokenizer(Path modelPath) {
        try {
            return TokenizerFactory.load(modelPath, null);
        } catch (IOException e) {
            log.errorf(e, "Tokenizer loading failed for [%s]", modelPath);
            throw new RuntimeException("Tokenizer loading failed: " + e.getMessage(), e);
        }
    }

    private ModelConfig loadConfig(Path modelPath) {
        try {
            Path configDir = Files.isRegularFile(modelPath) ? modelPath.getParent() : modelPath;
            if (configDir != null) {
                return ModelConfig.fromDirectory(configDir, objectMapper);
            }
            return new ModelConfig();
        } catch (IOException e) {
            log.warnf(e, "DirectInferenceEngine: failed to read config.json near [%s]", modelPath);
            return new ModelConfig();
        }
    }

    private static void applyLogitSoftcap(float[] logits, ModelConfig config) {
        if (logits == null)
            return;
        Double cap = config.finalLogitSoftcapping();
        if (cap == null || cap <= 0)
            return;
        float c = cap.floatValue();
        for (int i = 0; i < logits.length; i++) {
            float x = logits[i];
            logits[i] = (float) (c * Math.tanh(x / c));
        }
    }
}
