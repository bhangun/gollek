package tech.kayys.gollek.sdk.core;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.sdk.mcp.McpRegistryManager;
import tech.kayys.gollek.sdk.model.ModelInfo;
import tech.kayys.gollek.sdk.model.ModelResolution;
import tech.kayys.gollek.sdk.model.PullProgress;
import tech.kayys.gollek.sdk.model.SystemInfo;
import tech.kayys.gollek.spi.batch.BatchInferenceRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;
import tech.kayys.gollek.spi.inference.AsyncJobStatus;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.ModelRegistry;
import tech.kayys.gollek.spi.model.MultimodalRequest;
import tech.kayys.gollek.spi.model.MultimodalResponse;
import tech.kayys.gollek.spi.plugin.GollekPlugin;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModelFormat;

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

    Multi<StreamingInferenceChunk> streamCompletion(InferenceRequest request);

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
    default Multi<StreamingInferenceChunk> streamGguf(String modelId, InferenceRequest request) {
        return streamCompletion(request.toBuilder()
                .model(modelId)
                .preferredProvider("gguf")
                .streaming(true)
                .build());
    }

    /**
     * Stream tokens from a SafeTensors model.
     */
    default Multi<StreamingInferenceChunk> streamSafeTensors(String modelId, InferenceRequest request) {
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
    default ModelResolution prepareModel(String modelId, boolean forceGguf, Consumer<PullProgress> progressCallback) throws SdkException {
        return prepareModel(modelId, forceGguf, "Q4_K_M", progressCallback);
    }

    /**
     * Resolve and pull a model if necessary with quantization control.
     * 
     * @param modelId Model identifier
     * @param forceGguf Force GGUF format
     * @param quantization Quantization type (Q4_0, Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0, F16, F32)
     * @param progressCallback Progress callback
     * @return Model resolution
     */
    default ModelResolution prepareModel(String modelId, boolean forceGguf, String quantization, Consumer<PullProgress> progressCallback) throws SdkException {
        return new ModelResolution(modelId, null, getModelInfo(modelId).orElse(null));
    }

    /**
     * Explicitly convert a model (checkpoint/tensor) to GGUF format.
     * NEW in v1.2.3.
     */
    default ModelResolution convertToGguf(ModelResolution source, String quantization, Consumer<PullProgress> progressCallback) throws SdkException {
        throw new UnsupportedOperationException("GGUF conversion not supported by this SDK implementation.");
    }

    /**
     * Automatically select a provider for a model based on its format. NEW in
     * v1.2.1.
     */
    default Optional<String> autoSelectProvider(String modelId, boolean forceGguf) throws SdkException {
        return autoSelectProvider(modelId, forceGguf, "Q4_K_M");
    }

    /**
     * Automatically select a provider for a model based on its format with quantization.
     */
    default Optional<String> autoSelectProvider(String modelId, boolean forceGguf, String quantization) throws SdkException {
        return Optional.empty();
    }

    /**
     * Resolve, pull, and auto-select provider for a model. NEW in v1.2.1.
     */
    default ModelResolution ensureModelAvailable(String modelId, boolean forceGguf, Consumer<PullProgress> progressCallback)
            throws SdkException {
        return ensureModelAvailable(modelId, forceGguf, "Q4_K_M", progressCallback);
    }

    /**
     * Resolve, pull, and auto-select provider for a model with quantization control.
     */
    default ModelResolution ensureModelAvailable(String modelId, boolean forceGguf, String quantization, Consumer<PullProgress> progressCallback)
            throws SdkException {
        ModelResolution resolution = prepareModel(modelId, forceGguf, quantization, progressCallback);
        if (getPreferredProvider().isEmpty()) {
            Optional<String> autoProvider = autoSelectProvider(resolution.getModelId(), forceGguf, quantization);
            if (autoProvider.isPresent()) {
                setPreferredProvider(autoProvider.get());
            }
        }
        return resolution;
    }

    /**
     * Resolve a default model if none specified. NEW in v1.2.1.
     */
    default Optional<String> resolveDefaultModel() throws SdkException {
        return listModels(0, 1).stream().findFirst().map(ModelInfo::getModelId);
    }

    /**
     * Check if a provider is a cloud provider. NEW in v1.2.1.
     */
    default boolean isCloudProvider(String providerId) {
        if (providerId == null)
            return false;
        String p = providerId.toLowerCase();
        return p.equals("openai") || p.equals("mistral") || p.equals("anthropic") || p.equals("gemini")
                || p.equals("cerebras") || p.equals("deepseek") || p.equals("ollama");
    }

    /**
     * Check if a provider is an MCP tool provider. NEW in v1.2.1.
     */
    default boolean isMcpProvider(String providerId) {
        return "mcp".equalsIgnoreCase(providerId);
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

    /**
     * Pull a model with explicit revision and force options. NEW in v1.2.2.
     * 
     * @param modelSpec model name or hf:RepoId
     * @param revision branch, tag or commit hash (null for default)
     * @param force force re-download even if files exist
     * @param progressCallback progress updates
     */
    default void pullModel(String modelSpec, String revision, boolean force, Consumer<PullProgress> progressCallback) throws SdkException {
        // Default implementation for backward compatibility or implementations that don't support revision/force yet.
        // It's better to log a warning or just delegate if revision is null and force is false.
        if (revision == null && !force) {
            pullModel(modelSpec, progressCallback);
        } else {
            throw new SdkException("SDK_ERR_NOT_SUPPORTED", 
                "This SDK implementation does not support explicit revision or force pull.");
        }
    }

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

    // ==================== Multimodal Operations (v0.1.5) ====================

    /**
     * Process a multimodal request (image captioning, VQA, classification, embedding).
     * Routes to the appropriate multimodal processor based on model and task type.
     *
     * @param request the multimodal request containing inputs and task specification
     * @return multimodal response with outputs
     * @throws SdkException if processing fails
     */
    default MultimodalResponse processMultimodal(MultimodalRequest request) throws SdkException {
        throw new UnsupportedOperationException(
                "Multimodal processing is not supported by this SDK implementation. "
                + "Ensure gollek-multimodal-core is on the classpath.");
    }
}
