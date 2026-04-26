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
// import tech.kayys.gollek.safetensor.engine.warmup.DirectSafetensorBackend;

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

    // @Inject
    // DirectSafetensorBackend directSafetensorBackend;

    @Inject
    @Any
    Instance<StreamingProvider> availableProviders;

    @Inject
    tech.kayys.gollek.models.core.ModelArchitectureRegistry architectureRegistry;

    /**
     * Lazily resolves the LibTorch provider from the CDI container.
     * Returns null if it's not available on the classpath.
     */
    private StreamingProvider resolveLibTorchFallback() {
        if (availableProviders == null)
            return null;
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
                        tech.kayys.gollek.spi.model.ModelFormat.SAFETENSORS))
                .supportedDevices(java.util.Set.of(tech.kayys.gollek.spi.model.DeviceType.CPU)) // Extension handles
                                                                                                // acceleration if
                                                                                                // available
                .features(java.util.Set.of("native_jit", "ffm_acceleration"))
                .build();
    }

    @Override
    public void initialize(ProviderConfig config) {
        LOG.info("Gollek Native inference engine initialized");
        // if (directSafetensorBackend != null) {
        // LOG.info("Direct Safetensor backend detected and ready");
        // }
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
            if ("GGUF".equalsIgnoreCase(String.valueOf(format))
                    || "SAFETENSORS".equalsIgnoreCase(String.valueOf(format))
                    || "SAFETENSOR".equalsIgnoreCase(String.valueOf(format))) {
                return true;
            }

            // Check model_path in metadata
            Object path = request.getMetadata().get("model_path");
            if (path != null) {
                String pathStr = String.valueOf(path);
                String lowerPathStr = pathStr.toLowerCase();
                if (lowerPathStr.endsWith(".gguf") || lowerPathStr.endsWith(".safetensors")
                        || lowerPathStr.endsWith(".safetensor")) {
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
        if (modelPath == null)
            return false;
        String lowerPath = modelPath.toLowerCase();
        if (lowerPath.endsWith(".safetensors") || lowerPath.endsWith(".safetensor")) {
            return true;
        }
        return containsSafetensorFiles(modelPath);
    }

    /**
     * Checks if a given path is a directory containing .safetensors model files.
     * This handles the common case where the SDK resolves a model to a blob
     * directory
     * (e.g. ~/.gollek/models/blobs/UUID/) containing model.safetensors and
     * config.json.
     */
    private static boolean containsSafetensorFiles(String pathStr) {
        try {
            Path dir = Path.of(pathStr);
            if (!Files.isDirectory(dir))
                return false;
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
            throw new UnsupportedOperationException("Safetensor support is currently disabled in the native engine");
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
            throw new UnsupportedOperationException("Safetensor support is currently disabled in the native engine");
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
            String arch = (String) model.metadata().getOrDefault("general.architecture", "llama");
            if (arch.equals("gemma")) {
                Object aCap = model.metadata().get(arch + ".attention.logit_softcapping");
                int nLayers = 0;
                Object blockCountObj = model.metadata().get(arch + ".block_count");
                if (blockCountObj instanceof Number n)
                    nLayers = n.intValue();
                if (aCap != null || nLayers == 42 || nLayers == 26 || nLayers == 46
                        || model.tensors().stream().anyMatch(t -> t.name().contains("post_attention_layernorm"))) {
                    arch = "gemma2";
                }
                System.out.println("Heuristic: aCap=" + aCap + ", nLayers=" + nLayers + " -> arch=" + arch);
            }
            tech.kayys.gollek.spi.model.ModelArchitecture architecture = architectureRegistry.resolveGguf(arch);
            NativeInferenceEngine engine = new NativeInferenceEngine(model, architecture);
            engineCache.put(modelPath, engine);
        }
        return engineCache.get(modelPath);
    }

    private void executeInference(ProviderRequest request, MultiEmitter<? super StreamingInferenceChunk> emitter)
            throws Exception {
        String modelPath = request.getParameter("model_path", String.class)
                .or(() -> Optional.ofNullable((String) request.getMetadata().get("model_path")))
                .orElse(request.getModel());

        long loadStartTime = System.nanoTime();
        NativeInferenceEngine engine = getOrLoadEngine(modelPath);
        long loadEndTime = System.nanoTime();

        GGUFTokenizer tokenizer = new GGUFTokenizer(engine.getModel());

        // 1. Prepare prompt with ChatTemplate
        String arch = (String) engine.getModel().metadata().getOrDefault("general.architecture", "llama");
        System.out.println("Chat Template from Metadata: " + engine.getModel().metadata().get("tokenizer.chat_template"));
        System.out.println("Applying chat template...");
        ChatTemplate template = ChatTemplate.forArchitecture(arch);
        String promptText = template.apply(request.getMessages(), tokenizer.specialTokens());

        System.out.println("Encoding prompt...");
        long[] baseTokens = tokenizer.encode(promptText, EncodeOptions.builder().build());
        long[] tokens;
        boolean isGemma = arch.toLowerCase().contains("gemma");
        boolean shouldAddBos = isGemma;
        if (tokenizer instanceof GGUFTokenizer gt && gt.shouldAddBos()) {
            shouldAddBos = true;
        }

        if (shouldAddBos && tokenizer.bosTokenId() >= 0) {
            tokens = new long[baseTokens.length + 1];
            tokens[0] = tokenizer.bosTokenId();
            System.arraycopy(baseTokens, 0, tokens, 1, baseTokens.length);
        } else {
            tokens = baseTokens;
        }
        System.out.println("Encoded " + tokens.length + " tokens.");
        for (int i = 0; i < tokens.length; i++) {
            System.out.println("Token [" + i + "]: " + tokens[i] + " -> '" + tokenizer.decode(new long[]{tokens[i]}, null) + "'");
        }

        try (NativeInferenceSession session = new NativeInferenceSession(engine, 4096)) {
            System.out.println("Starting optimized batch pre-fill...");
            long prefillStartTime = System.nanoTime();

            // 1. Embedding lookup for all prompt tokens
            int nPrompt = tokens.length - 1;
            MemorySegment[] xBatch = new MemorySegment[nPrompt];
            for (int t = 0; t < nPrompt; t++) {
                xBatch[t] = session.getArena().allocate(engine.getHidden() * 4, 64);
                session.lookupEmbedding((int) tokens[t], xBatch[t]);
                float scale = engine.getArchitecture().embeddingScaleFactor(engine.getHidden());
                if (scale != 1.0f) {
                    for (int j = 0; j < engine.getHidden(); j++) {
                        float v = xBatch[t].get(java.lang.foreign.ValueLayout.JAVA_FLOAT, (long) j * 4);
                        xBatch[t].set(java.lang.foreign.ValueLayout.JAVA_FLOAT, (long) j * 4, v * scale);
                    }
                }
            }

            // 2. Process layer by layer
            int ffnDim = engine.getFfnDim();
            for (int i = 0; i < engine.getNLayers(); i++) {
                System.out.print("\rPre-filling layer " + i + "/" + engine.getNLayers() + "... ");
                System.out.flush();

                // Dequantize this layer's weights ONCE for the whole batch
                try (java.lang.foreign.Arena layerArena = java.lang.foreign.Arena.ofConfined()) {
                    tech.kayys.gollek.spi.tensor.weights.TransformerLayerWeights dequantWeights = engine.getLayers().get(i)
                            .dequantize(layerArena);

                    // Process all tokens for this layer
                    for (int t = 0; t < nPrompt; t++) {
                        FusedTransformerLayer.execute(
                                i, t, xBatch[t], xBatch[t], // use same buffer for in/out
                                dequantWeights, session.getLayerBuffers().get(i), session.getKvCache(),
                                engine.getRopeCache(), engine.getHidden(), engine.getNHeads(),
                                engine.getNHeadsKv(), engine.getHeadDim(), engine.isNeox(),
                                engine.getEps(), engine.getAttnSoftCap(),
                                engine.getArchitecture().addOneToRmsNormWeight(),
                                engine.getArchitecture().activationType(),
                                engine.getNumExperts(),
                                engine.getNumExpertsPerTok(),
                                engine.getMetrics(),
                                executor);
                    }
                }
            }
            System.out.println("\nPre-fill complete.");

            // 3. Last prompt token goes to the session residual 'x'
            session.setPos(nPrompt);
            session.lookupEmbedding((int) tokens[tokens.length - 1], session.getX());

            long prefillEndTime = System.nanoTime();
            engine.getMetrics().recordPrompt(prefillEndTime - prefillStartTime);

            long firstTokenTime = 0;

            int lastToken = (int) tokens[tokens.length - 1];
            int maxNewTokens = request.getMaxTokens() > 0 ? request.getMaxTokens() : 512;

            System.out.println("Starting generation loop...");

            // Extract sampling parameters using standardized helper methods
            float temperature = request.getTemperature() > 0 ? (float) request.getTemperature() : 1.0f;
            int topK = request.getTopK() > 0 ? request.getTopK() : 40;
            float topP = request.getTopP() > 0 ? (float) request.getTopP() : 0.9f;
            float repPenalty = request.getRepeatPenalty() > 0 ? (float) request.getRepeatPenalty() : 1.1f;

            // Handle greedy decoding explicitly if temperature is set to exactly 0 in
            // request
            if (request.getTemperature() == 0.0)
                temperature = 0.0f;

            List<Integer> tokenHistory = new ArrayList<>();
            for (long t : tokens)
                tokenHistory.add((int) t);

            int generatedCount = 0;
            // Robust vocab size lookup
            int vocabSize = engine.getVocabSize();

            try (Arena localArena = Arena.ofAuto()) {
                MemorySegment logits = localArena.allocate((long) vocabSize * Float.BYTES, 64);

                for (int i = 0; i < maxNewTokens; i++) {
                    long tokenStartTime = System.nanoTime();
                    MemorySegment hidden = session.tick(lastToken, executor);

                    if (i == 0)
                        firstTokenTime = System.nanoTime();
                    
                    long tokenEndTime = System.nanoTime();
                    engine.getMetrics().recordToken(tokenEndTime - tokenStartTime);

                    // Logit Projection
                    LogitProjectionKernel.execute(hidden, engine.getOutputWeight(), logits, engine.getHidden(),
                            vocabSize, engine.getFinalSoftCap(), executor);

                    // Controlled Sampling
                    lastToken = SamplingKernel.sample(logits, vocabSize, temperature, topK, topP, repPenalty,
                            tokenHistory);
                    tokenHistory.add(lastToken);
                    generatedCount++;

                    if (tokenizer.isEosToken(lastToken))
                        break;

                    String delta = tokenizer.decode(new long[] { lastToken }, null);
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

            if (engine.getNumExperts() > 0) {
                Map<String, Double> expertUsage = new HashMap<>();
                for (int j = 0; j < engine.getNumExperts(); j++) {
                    double util = engine.getMetrics().getExpertUtilization(j);
                    if (util > 0) {
                        expertUsage.put("expert_" + j, util);
                        LOG.infof("Expert %d utilization: %.2f%%", j, util * 100);
                    }
                }
                metrics.put("moe.expert_utilization", expertUsage);
            }

            emitter.emit(StreamingInferenceChunk.withMetadata(
                    request.getRequestId(),
                    generatedCount,
                    "",
                    metrics));

            LOG.infof("Inference Bench: Prefill %.2f t/s, Gen %.2f t/s, TTFT %.2f ms", prefillTps, genTps,
                    (firstTokenTime - prefillStartTime) / 1_000_000.0);

            emitter.complete();
        }
    }
}
