package tech.kayys.gollek.inference.nativeimpl;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.subscription.MultiEmitter;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Any;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.tokenizer.GGUFTokenizer;
import tech.kayys.gollek.gguf.loader.GGUFLoader;
import tech.kayys.gollek.spi.inference.*;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.safetensor.engine.warmup.DirectSafetensorBackend;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@ApplicationScoped
public class NativeLLMProvider implements StreamingProvider {

    private static final Logger LOG = Logger.getLogger(NativeLLMProvider.class);

    @Inject
    DirectSafetensorBackend directSafetensorBackend;

    @Inject
    @Any
    Instance<StreamingProvider> availableProviders;

    /**
     * Lazily resolves the LibTorch provider from the CDI container.
     * Returns null if it's not available on the classpath.
     */
    private StreamingProvider resolveLibTorchFallback() {
        if (availableProviders == null) return null;
        for (StreamingProvider p : availableProviders) {
            if ("libtorch".equalsIgnoreCase(p.id())) {
                return p;
            }
        }
        return null;
    }

    @Override
    public String id() {
        return "native";
    }

    @Override
    public String name() {
        return "Native LLM Provider";
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId("native")
                .name("Gollek Native Engine")
                .version("1.0.0")
                .description("Pure Java inference engine for GGUF and Safetensor models using FFM and Vector API")
                .vendor("GollekPlatform")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .supportedFormats(java.util.Set.of(
                        tech.kayys.gollek.spi.model.ModelFormat.GGUF,
                        tech.kayys.gollek.spi.model.ModelFormat.SAFETENSORS
                ))
                .supportedDevices(java.util.Set.of(tech.kayys.gollek.spi.model.DeviceType.CPU)) // Extension handles acceleration if available
                .features(java.util.Set.of("native_jit", "ffm_acceleration"))
                .build();
    }

    @Override
    public void initialize(ProviderConfig config) {
        LOG.info("Gollek Native inference engine initialized");
        if (directSafetensorBackend != null) {
            directSafetensorBackend.initialize(config);
        }
        StreamingProvider libtorch = resolveLibTorchFallback();
        if (libtorch != null) {
            try {
                libtorch.initialize(config);
            } catch (Exception e) {
                LOG.warn("Failed to initialize LibTorch fallback provider", e);
            }
        }
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        if (modelId == null) {
            return false;
        }

        // 1. Explicit preference for native engine
        if (request != null && "native".equalsIgnoreCase(request.getPreferredProvider().orElse(null))) {
            return true;
        }

        // 2. Explicit GGUF or Safetensor requested via parameter
        if (request != null && request.getParameter("gguf", Boolean.class).orElse(false)) {
            return true;
        }

        // 3. Extension check
        String lowerId = modelId.toLowerCase();
        if (lowerId.endsWith(".gguf") || lowerId.endsWith(".safetensors") || lowerId.endsWith(".safetensor")) {
            return true;
        }

        // 3b. Directory containing safetensor files
        if (containsSafetensorFiles(modelId)) {
            return true;
        }

        // 4. Metadata check for format
        if (request != null) {
            Object format = request.getMetadata().get("format");
            if ("GGUF".equalsIgnoreCase(String.valueOf(format)) || "SAFETENSORS".equalsIgnoreCase(String.valueOf(format)) || "SAFETENSOR".equalsIgnoreCase(String.valueOf(format))) {
                return true;
            }
            
            // Check model_path in metadata
            Object path = request.getMetadata().get("model_path");
            if (path != null) {
                String pathStr = String.valueOf(path);
                String lowerPathStr = pathStr.toLowerCase();
                if (lowerPathStr.endsWith(".gguf") || lowerPathStr.endsWith(".safetensors") || lowerPathStr.endsWith(".safetensor")) {
                    return true;
                }
                if (containsSafetensorFiles(pathStr)) {
                    return true;
                }
            }
        }

        return modelId.contains("native");
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(ProviderHealth.healthy("Native engine is ready"));
    }

    @Override
    public void shutdown() {
        executor.shutdown();
        StreamingProvider libtorch = resolveLibTorchFallback();
        if (libtorch != null) {
            libtorch.shutdown();
        }
    }

    // Shared executor for parallel kernel operations
    private final ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    
    // Simple cache for engine instances
    private final Map<String, NativeInferenceEngine> engineCache = new HashMap<>();

    private boolean isSafetensorModel(ProviderRequest request) {
        String modelPath = request.getParameter("model_path", String.class)
                .or(() -> Optional.ofNullable((String) request.getMetadata().get("model_path")))
                .orElse(request.getModel());
        if (modelPath == null) return false;
        String lowerPath = modelPath.toLowerCase();
        if (lowerPath.endsWith(".safetensors") || lowerPath.endsWith(".safetensor")) {
            return true;
        }
        return containsSafetensorFiles(modelPath);
    }

    /**
     * Checks if a given path is a directory containing .safetensors model files.
     * This handles the common case where the SDK resolves a model to a blob directory
     * (e.g. ~/.gollek/models/blobs/UUID/) containing model.safetensors and config.json.
     */
    private static boolean containsSafetensorFiles(String pathStr) {
        try {
            Path dir = Path.of(pathStr);
            if (!Files.isDirectory(dir)) return false;
            try (var stream = Files.list(dir)) {
                return stream.anyMatch(f -> {
                    String name = f.getFileName().toString().toLowerCase();
                    return name.endsWith(".safetensors") || name.endsWith(".safetensor");
                });
            }
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Creates a new ProviderRequest with the model field set to the resolved
     * physical path from model_path parameter/metadata. This ensures downstream
     * backends (DirectSafetensorBackend, LibTorch) receive an absolute filesystem
     * path rather than an abstract model ID like "HuggingFaceTB/SmolLM2-360M".
     */
    private ProviderRequest requestWithResolvedModelPath(ProviderRequest request) {
        String resolvedPath = request.getParameter("model_path", String.class)
                .or(() -> Optional.ofNullable((String) request.getMetadata().get("model_path")))
                .orElse(request.getModel());
        return new ProviderRequest(
                request.getRequestId(),
                resolvedPath,
                request.getMessages(),
                request.getParameters(),
                request.getTools(),
                request.getToolChoice(),
                request.isStreaming(),
                request.getTimeout(),
                request.getUserId().orElse(null),
                request.getSessionId().orElse(null),
                request.getTraceId().orElse(null),
                request.getApiKey().orElse(null),
                request.getMetadata(),
                request.getPreferredProvider().orElse(null));
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        if (isSafetensorModel(request)) {
            LOG.info("Native engine delegating Safetensor evaluation to DirectSafetensorBackend");
            ProviderRequest resolved = requestWithResolvedModelPath(request);
            return directSafetensorBackend.infer(resolved)
                    .onFailure().recoverWithUni(failure -> {
                        LOG.warn("DirectSafetensorBackend failed, attempting LibTorch fallback", failure);
                        StreamingProvider libtorch = resolveLibTorchFallback();
                        if (libtorch != null) {
                            return libtorch.infer(resolved);
                        }
                        return Uni.createFrom().failure(failure);
                    });
        }

        return inferStream(request)
                .onItem().transform(StreamingInferenceChunk::getDelta)
                .filter(java.util.Objects::nonNull)
                .collect().in(StringBuilder::new, StringBuilder::append)
                .onItem().transform(sb -> {
                    return InferenceResponse.builder()
                            .requestId(request.getRequestId())
                            .model(request.getModel())
                            .content(sb.toString())
                            .build();
                });
    }

    @Override
    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        if (isSafetensorModel(request)) {
            LOG.info("Native engine delegating streaming Safetensor evaluation to DirectSafetensorBackend");
            ProviderRequest resolved = requestWithResolvedModelPath(request);
            return directSafetensorBackend.inferStream(resolved)
                    .onFailure().recoverWithMulti(failure -> {
                        LOG.warn("DirectSafetensorBackend failed, attempting LibTorch fallback", failure);
                        StreamingProvider libtorch = resolveLibTorchFallback();
                        if (libtorch != null) {
                            return libtorch.inferStream(resolved);
                        }
                        return Multi.createFrom().failure(failure);
                    });
        }

        return Multi.createFrom().emitter(emitter -> {
            try {
                executeInference(request, emitter);
            } catch (Exception e) {
                LOG.error("Inference failed", e);
                emitter.fail(e);
            }
        });
    }

    private synchronized NativeInferenceEngine getOrLoadEngine(String modelPath) throws Exception {
        if (!engineCache.containsKey(modelPath)) {
            GGUFModel model = GGUFLoader.loadModel(Path.of(modelPath));
            NativeInferenceEngine engine = new NativeInferenceEngine(model);
            engineCache.put(modelPath, engine);
        }
        return engineCache.get(modelPath);
    }

    private void executeInference(ProviderRequest request, MultiEmitter<? super StreamingInferenceChunk> emitter) throws Exception {
        String modelPath = request.getParameter("model_path", String.class)
                .or(() -> Optional.ofNullable((String) request.getMetadata().get("model_path")))
                .orElse(request.getModel());

        long loadStartTime = System.nanoTime();
        NativeInferenceEngine engine = getOrLoadEngine(modelPath);
        long loadEndTime = System.nanoTime();
        
        GGUFTokenizer tokenizer = new GGUFTokenizer(engine.getModel());

        // 1. Prepare prompt with ChatTemplate
        String arch = (String) engine.getModel().metadata().getOrDefault("general.architecture", "llama");
        ChatTemplate template = ChatTemplate.forArchitecture(arch);
        String promptText = template.apply(request.getMessages());

        long[] tokens = tokenizer.encode(promptText, EncodeOptions.builder().build());
        
        LOG.debugf("Prompt Tokens: %d", tokens.length);
        
        try (NativeInferenceSession session = new NativeInferenceSession(engine, 4096)) {
            long prefillStartTime = System.nanoTime();
            
            // Prefill loop (process all but the last prompt token)
            for (int i = 0; i < tokens.length - 1; i++) {
                session.tick((int) tokens[i], executor);
            }

            long prefillEndTime = System.nanoTime();
            long firstTokenTime = 0;
            
            int lastToken = (int) tokens[tokens.length - 1];
            int maxNewTokens = request.getMaxTokens() > 0 ? request.getMaxTokens() : 512;
            
            // Extract sampling parameters using standardized helper methods
            float temperature = request.getTemperature() > 0 ? (float) request.getTemperature() : 1.0f;
            int topK = request.getTopK() > 0 ? request.getTopK() : 40;
            float topP = request.getTopP() > 0 ? (float) request.getTopP() : 0.9f;
            float repPenalty = request.getRepeatPenalty() > 0 ? (float) request.getRepeatPenalty() : 1.1f;

            // Handle greedy decoding explicitly if temperature is set to exactly 0 in request
            if (request.getTemperature() == 0.0) temperature = 0.0f;

            List<Integer> tokenHistory = new ArrayList<>();
            for (long t : tokens) tokenHistory.add((int) t);

            int generatedCount = 0;
            // Robust vocab size lookup
            int vocabSize = engine.getVocabSize();

            try (Arena localArena = Arena.ofShared()) {
                MemorySegment logits = localArena.allocate((long) vocabSize * Float.BYTES, 64);

                for (int i = 0; i < maxNewTokens; i++) {
                    MemorySegment hidden = session.tick(lastToken, executor);
                    
                    if (i == 0) firstTokenTime = System.nanoTime();

                    // Logit Projection
                    LogitProjectionKernel.execute(hidden, engine.getOutputWeight(), logits, engine.getHidden(), vocabSize);
                    
                    // Controlled Sampling
                    lastToken = SamplingKernel.sample(logits, vocabSize, temperature, topK, topP, repPenalty, tokenHistory);
                    tokenHistory.add(lastToken);
                    generatedCount++;

                    if (tokenizer.isEosToken(lastToken)) break;

                    String delta = tokenizer.decode(new long[]{lastToken}, null);
                    emitter.emit(StreamingInferenceChunk.of(request.getRequestId(), i, delta));
                }
            }

            long generationEndTime = System.nanoTime();
            
            // 2. Wrap up metrics
            Map<String, Object> metrics = new HashMap<>();
            metrics.put("bench.load_ms", (loadEndTime - loadStartTime) / 1_000_000.0);
            metrics.put("bench.prefill_ms", (prefillEndTime - prefillStartTime) / 1_000_000.0);
            metrics.put("bench.decode_ms", (generationEndTime - prefillEndTime) / 1_000_000.0);
            metrics.put("bench.ttft_ms", (firstTokenTime - prefillStartTime) / 1_000_000.0);
            
            double prefillTps = (tokens.length - 1) / ((prefillEndTime - prefillStartTime) / 1_000_000_000.0);
            double genTps = generatedCount / ((generationEndTime - prefillEndTime) / 1_000_000_000.0);
            
            metrics.put("bench.prefill_tps", prefillTps);
            metrics.put("bench.generation_tps", genTps);
            metrics.put("tokens.input", tokens.length);
            metrics.put("tokens.output", generatedCount);

            emitter.emit(StreamingInferenceChunk.withMetadata(
                request.getRequestId(), 
                generatedCount, 
                "", 
                metrics
            ));
            
            LOG.infof("Inference Bench: Prefill %.2f t/s, Gen %.2f t/s, TTFT %.2f ms", prefillTps, genTps, (firstTokenTime - prefillStartTime) / 1_000_000.0);
            
            emitter.complete();
        }
    }
}
