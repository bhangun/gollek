package tech.kayys.gollek.sdk.core;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.sdk.mcp.McpRegistryManager;
import tech.kayys.gollek.sdk.model.ModelInfo;
import tech.kayys.gollek.sdk.model.ModelResolution;
import tech.kayys.gollek.sdk.model.PullProgress;
import tech.kayys.gollek.sdk.model.SystemInfo;
import tech.kayys.gollek.spi.inference.AsyncJobStatus;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.ModelRegistry;
import tech.kayys.gollek.spi.plugin.GollekPlugin;
import tech.kayys.gollek.spi.inference.BatchInferenceRequest;
import tech.kayys.gollek.spi.inference.EmbeddingRequest;
import tech.kayys.gollek.spi.inference.EmbeddingResponse;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.spi.stream.StreamChunk;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Core interface for the Gollek inference engine SDK — v0.1.4.
 *
 * <h3>New in v0.1.4</h3>
 * <ul>
 * <li>{@link #listModelsByFormat} — filter the model catalogue by format.</li>
 * <li>{@link #inferGguf} / {@link #inferSafeTensors} — convenience methods
 * that pre-set the format hint, skipping runtime detection.</li>
 * <li>{@link #streamGguf} / {@link #streamSafeTensors} — streaming
 * variants.</li>
 * <li>{@link #embed} — embedding shortcut with a plain string input.</li>
 * <li>{@link #pullGgufModel} / {@link #pullSafeTensorsModel} — format-aware
 * model pull helpers.</li>
 * <li>{@link #getSystemInfo()} promoted to non-default.</li>
 * </ul>
 */
public interface GollekSdk {

    // ==================== Core Inference ====================

    InferenceResponse createCompletion(InferenceRequest request) throws SdkException;

    java.util.concurrent.CompletableFuture<InferenceResponse> createCompletionAsync(InferenceRequest request);

    Multi<StreamChunk> streamCompletion(InferenceRequest request);

    // ==================== Format-Aware Inference (v0.1.4) ====================

    /**
     * Run inference on a GGUF model. Equivalent to setting
     * {@code request.preferredProvider = "gguf"} and calling
     * {@link #createCompletion}.
     *
     * @param modelId bare name, filename stem, or absolute path of the GGUF file
     * @param request inference request (model field is overridden by
     *                {@code modelId})
     */
    default InferenceResponse inferGguf(String modelId, InferenceRequest request) throws SdkException {
        return createCompletion(request.toBuilder()
                .model(modelId)
                .preferredProvider("gguf")
                .build());
    }

    /**
     * Run inference on a SafeTensors checkpoint.
     *
     * @param modelId model directory name, alias, or absolute path
     * @param request inference request
     */
    default InferenceResponse inferSafeTensors(String modelId, InferenceRequest request) throws SdkException {
        return createCompletion(request.toBuilder()
                .model(modelId)
                .preferredProvider("safetensor")
                .build());
    }

    /**
     * Stream tokens from a GGUF model.
     */
    default Multi<StreamChunk> streamGguf(String modelId, InferenceRequest request) {
        return streamCompletion(request.toBuilder()
                .model(modelId)
                .preferredProvider("gguf")
                .streaming(true)
                .build());
    }

    /**
     * Stream tokens from a SafeTensors model.
     */
    default Multi<StreamChunk> streamSafeTensors(String modelId, InferenceRequest request) {
        return streamCompletion(request.toBuilder()
                .model(modelId)
                .preferredProvider("safetensor")
                .streaming(true)
                .build());
    }

    // ==================== Embeddings ====================

    EmbeddingResponse createEmbedding(EmbeddingRequest request) throws SdkException;

    /**
     * Convenience embedding — single text string, auto-routes to whatever
     * provider handles {@code modelId}. NEW in v0.1.4.
     *
     * @param modelId model identifier (GGUF or SafeTensors)
     * @param text    text to embed
     */
    default EmbeddingResponse embed(String modelId, String text) throws SdkException {
        return createEmbedding(EmbeddingRequest.builder()
                .model(modelId)
                .input(text)
                .build());
    }

    // ==================== Async Jobs ====================

    String submitAsyncJob(InferenceRequest request) throws SdkException;

    AsyncJobStatus getJobStatus(String jobId) throws SdkException;

    AsyncJobStatus waitForJob(String jobId, java.time.Duration maxWaitTime, java.time.Duration pollInterval)
            throws SdkException;

    List<InferenceResponse> batchInference(BatchInferenceRequest batchRequest) throws SdkException;

    // ==================== Provider Operations ====================

    List<ProviderInfo> listAvailableProviders() throws SdkException;

    ProviderInfo getProviderInfo(String providerId) throws SdkException;

    void setPreferredProvider(String providerId) throws SdkException;

    Optional<String> getPreferredProvider();

    // ==================== Model Operations ====================

    /**
     * Resolve and pull a model if necessary. NEW in v1.2.1.
     */
    default ModelResolution prepareModel(String modelId, Consumer<PullProgress> progressCallback) throws SdkException {
        return new ModelResolution(modelId, null, getModelInfo(modelId).orElse(null));
    }

    /**
     * Automatically select a provider for a model based on its format. NEW in v1.2.1.
     */
    default Optional<String> autoSelectProvider(String modelId) throws SdkException {
        return Optional.empty();
    }

    /**
     * Resolve a default model if none specified. NEW in v1.2.1.
     */
    default Optional<String> resolveDefaultModel() throws SdkException {
        return listModels(0, 1).stream().findFirst().map(ModelInfo::getModelId);
    }

    List<ModelInfo> listModels() throws SdkException;

    List<ModelInfo> listModels(int offset, int limit) throws SdkException;

    /**
     * List models filtered by format — NEW in v0.1.4.
     *
     * @param format GGUF, SAFETENSORS, etc. {@code null} = all formats.
     */
    default List<ModelInfo> listModelsByFormat(ModelFormat format) throws SdkException {
        return listModels().stream()
                .filter(m -> format == null || format.name().equalsIgnoreCase(m.getFormat()))
                .toList();
    }

    Optional<ModelInfo> getModelInfo(String modelId) throws SdkException;

    void pullModel(String modelSpec, Consumer<PullProgress> progressCallback) throws SdkException;

    void deleteModel(String modelId) throws SdkException;

    /**
     * Pull a GGUF model by name or HuggingFace repo. NEW in v0.1.4.
     *
     * @param modelSpec e.g. {@code "hf:TheBloke/Llama-2-7B-GGUF"} or
     *                  {@code "tinyllama"} (resolved via Ollama registry)
     * @param callback  progress callback (nullable)
     */
    default void pullGgufModel(String modelSpec, Consumer<PullProgress> callback) throws SdkException {
        pullModel(modelSpec.startsWith("gguf:") ? modelSpec : "gguf:" + modelSpec, callback);
    }

    /**
     * Pull a SafeTensors checkpoint from HuggingFace. NEW in v0.1.4.
     *
     * @param repoId   HuggingFace repo id, e.g. {@code "bert-base-uncased"}
     * @param callback progress callback (nullable)
     */
    default void pullSafeTensorsModel(String repoId, Consumer<PullProgress> callback) throws SdkException {
        pullModel("hf:" + repoId, callback);
    }

    // ==================== MCP Operations ====================

    default McpRegistryManager mcpRegistry() {
        throw new UnsupportedOperationException(
                "MCP registry is not supported by this SDK implementation");
    }

    // ==================== System Operations ====================

    SystemInfo getSystemInfo() throws SdkException;

    /**
     * Retrieve recent logs. NEW in v1.2.1.
     */
    default List<String> getRecentLogs(int maxLines) throws SdkException {
        return List.of();
    }

    // ==================== Advanced Operations (v1.2.2) ====================

    /**
     * List all installed plugins and their metadata.
     */
    default List<GollekPlugin.PluginMetadata> listPlugins() throws SdkException {
        return List.of();
    }

    /**
     * Get real-time performance metrics for a provider/model pair.
     * 
     * @return Map containing latency, error rate, etc.
     */
    default Map<String, Object> getMetrics(String providerId, String modelId) throws SdkException {
        return Map.of();
    }

    /**
     * Get detailed statistics for a model (usage, versions, etc.)
     */
    default Optional<ModelRegistry.ModelStats> getModelStats(String modelId) throws SdkException {
        return Optional.empty();
    }

    /**
     * Register a new model programmatically.
     */
    default ModelManifest registerModel(ModelRegistry.ModelUploadRequest request) throws SdkException {
        throw new UnsupportedOperationException("Model registration not supported by this SDK implementation");
    }
}
