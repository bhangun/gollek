package tech.kayys.gollek.safetensor.engine.sdk;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.sdk.model.ModelInfo;
import tech.kayys.gollek.sdk.model.ModelResolution;
import tech.kayys.gollek.sdk.model.PullProgress;
import tech.kayys.gollek.sdk.model.SystemInfo;
import tech.kayys.gollek.spi.batch.BatchInferenceRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;
import tech.kayys.gollek.spi.inference.AsyncJobStatus;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.sdk.core.GollekSdk;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * GollekSdk implementation backed by the SafeTensor DirectInferenceEngine.
 *
 * <p>This bridges the high-level SDK API (used by NLP pipelines, Gollek.java facade,
 * and the GollekClient builder) with the actual SafeTensor inference pipeline.
 *
 * <p>When a user calls:
 * <pre>{@code
 * var gen = Gollek.pipeline("text-generation", "Qwen/Qwen2.5-0.5B-Instruct");
 * String result = gen.process("Tell me about Java");
 * }</pre>
 * The call chain is:
 * <ol>
 *   <li>TextGenerationPipeline → GollekSdk.createCompletion()</li>
 *   <li>SafetensorGollekSdk.createCompletion() → DirectInferenceEngine.generate()</li>
 * </ol>
 */
public class SafetensorGollekSdk implements GollekSdk {

    private static final Logger log = Logger.getLogger(SafetensorGollekSdk.class);

    private final DirectInferenceEngine engine;
    private final Path modelBasePath;
    private String preferredProvider = "safetensor";

    /**
     * Creates a new SafetensorGollekSdk with the given inference engine.
     *
     * @param engine        the direct inference engine (may be CDI-managed or standalone)
     * @param modelBasePath base directory for model discovery
     */
    public SafetensorGollekSdk(DirectInferenceEngine engine, Path modelBasePath) {
        this.engine = engine;
        this.modelBasePath = modelBasePath;
        log.infof("SafetensorGollekSdk initialized with model base path [%s]", modelBasePath);
    }

    // ==================== Core Inference ====================

    @Override
    public InferenceResponse createCompletion(InferenceRequest request) throws SdkException {
        try {
            Path modelPath = resolveModelPath(request.getModel());
            GenerationConfig cfg = toGenerationConfig(request);

            Uni<InferenceResponse> uni = engine.generate(request.getPrompt(), modelPath, cfg);
            return uni.await().atMost(Duration.ofMinutes(5));
        } catch (Exception e) {
            throw new SdkException("SafeTensor inference failed: " + e.getMessage(), e);
        }
    }

    @Override
    public java.util.concurrent.CompletableFuture<InferenceResponse> createCompletionAsync(InferenceRequest request) {
        return java.util.concurrent.CompletableFuture.supplyAsync(() -> {
            try {
                return createCompletion(request);
            } catch (SdkException e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Override
    public Multi<StreamingInferenceChunk> streamCompletion(InferenceRequest request) {
        try {
            Path modelPath = resolveModelPath(request.getModel());
            GenerationConfig cfg = toGenerationConfig(request);
            String requestId = request.getRequestId() != null
                    ? request.getRequestId()
                    : java.util.UUID.randomUUID().toString();

            AtomicInteger index = new AtomicInteger(0);
            return engine.generateStream(request.getPrompt(), modelPath, cfg)
                    .map(response -> {
                        int idx = index.getAndIncrement();
                        boolean isFinal = response.getFinishReason() != null;
                        if (isFinal) {
                            return StreamingInferenceChunk.finalChunk(
                                    requestId, idx, response.getContent());
                        }
                        return StreamingInferenceChunk.textDelta(
                                requestId, idx, response.getContent());
                    });
        } catch (Exception e) {
            return Multi.createFrom().failure(
                    new SdkException("Streaming failed: " + e.getMessage(), e));
        }
    }

    // ==================== Embeddings ====================

    @Override
    public EmbeddingResponse createEmbedding(EmbeddingRequest request) throws SdkException {
        throw new SdkException("Embedding extraction is not yet supported by the SafeTensor provider. " +
                "Use a cloud provider or ONNX runtime for embeddings.");
    }

    // ==================== Async Jobs ====================

    @Override
    public String submitAsyncJob(InferenceRequest request) throws SdkException {
        createCompletionAsync(request);
        return java.util.UUID.randomUUID().toString();
    }

    @Override
    public AsyncJobStatus getJobStatus(String jobId) throws SdkException {
        return new AsyncJobStatus(jobId, null, null, "UNKNOWN", null, null, null, null);
    }

    @Override
    public AsyncJobStatus waitForJob(String jobId, Duration maxWaitTime, Duration pollInterval) throws SdkException {
        return getJobStatus(jobId);
    }

    @Override
    public List<InferenceResponse> batchInference(BatchInferenceRequest batchRequest) throws SdkException {
        return batchRequest.requests().stream()
                .map(req -> {
                    try { return createCompletion(req); }
                    catch (SdkException e) { throw new RuntimeException(e); }
                })
                .toList();
    }

    // ==================== Provider Operations ====================

    @Override
    public List<ProviderInfo> listAvailableProviders() throws SdkException {
        return List.of(ProviderInfo.builder()
                .id("safetensor")
                .name("SafeTensor Direct Inference")
                .description("Zero-copy SafeTensor model loading and inference via LibTorch FFM")
                .version("0.1.0")
                .build());
    }

    @Override
    public ProviderInfo getProviderInfo(String providerId) throws SdkException {
        return listAvailableProviders().getFirst();
    }

    @Override
    public void setPreferredProvider(String providerId) throws SdkException {
        this.preferredProvider = providerId;
    }

    @Override
    public Optional<String> getPreferredProvider() {
        return Optional.of(preferredProvider);
    }

    // ==================== Model Operations ====================

    @Override
    public List<ModelInfo> listModels() throws SdkException {
        return engine.listLoadedModels().stream()
                .map(m -> ModelInfo.builder()
                        .modelId(m.key())
                        .name(m.key())
                        .format("SAFETENSORS")
                        .build())
                .toList();
    }

    @Override
    public List<ModelInfo> listModels(int offset, int limit) throws SdkException {
        var all = listModels();
        return all.subList(Math.min(offset, all.size()), Math.min(offset + limit, all.size()));
    }

    @Override
    public Optional<ModelInfo> getModelInfo(String modelId) throws SdkException {
        return listModels().stream()
                .filter(m -> m.getModelId().equals(modelId))
                .findFirst();
    }

    @Override
    public void pullModel(String modelSpec, Consumer<PullProgress> progressCallback) throws SdkException {
        Path modelPath = resolveModelPath(modelSpec);
        if (modelPath == null || !Files.exists(modelPath)) {
            throw new SdkException("Model not found locally: " + modelSpec +
                    ". SafeTensor provider requires pre-downloaded models in " + modelBasePath);
        }
        log.infof("Model [%s] already available at [%s]", modelSpec, modelPath);
    }

    @Override
    public void deleteModel(String modelId) throws SdkException {
        Path modelPath = resolveModelPath(modelId);
        if (modelPath != null) {
            engine.unloadModel(modelPath);
        }
    }

    // ==================== System Operations ====================

    @Override
    public SystemInfo getSystemInfo() throws SdkException {
        return SystemInfo.builder()
                .cliVersion("0.1.0")
                .javaVersion(System.getProperty("java.version"))
                .osName(System.getProperty("os.name"))
                .osVersion(System.getProperty("os.version"))
                .osArch(System.getProperty("os.arch"))
                .build();
    }

    // ==================== Internal helpers ====================

    private Path resolveModelPath(String modelId) {
        if (modelId == null) return null;

        // 1. Absolute path
        Path direct = Path.of(modelId);
        if (Files.exists(direct)) return direct;

        // 2. Relative to model base path
        Path relative = modelBasePath.resolve(modelId);
        if (Files.exists(relative)) return relative;

        // 3. HuggingFace-style with separators normalized
        String normalized = modelId.replace("/", "--");
        Path hfStyle = modelBasePath.resolve(normalized);
        if (Files.exists(hfStyle)) return hfStyle;

        // 4. Check ~/.gollek/models
        Path gollekModels = Path.of(System.getProperty("user.home"), ".gollek", "models");
        Path gollekPath = gollekModels.resolve(normalized);
        if (Files.exists(gollekPath)) return gollekPath;

        return relative;
    }

    private GenerationConfig toGenerationConfig(InferenceRequest request) {
        Map<String, Object> params = request.getParameters();
        int maxTokens = params.containsKey("max_tokens")
                ? ((Number) params.get("max_tokens")).intValue() : 256;
        float temperature = params.containsKey("temperature")
                ? ((Number) params.get("temperature")).floatValue() : 0.7f;
        float topP = params.containsKey("top_p")
                ? ((Number) params.get("top_p")).floatValue() : 0.9f;

        return GenerationConfig.builder()
                .maxNewTokens(maxTokens)
                .temperature(temperature)
                .topP(topP)
                .build();
    }
}
