package tech.kayys.gollek.sdk.core;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.spi.stream.StreamChunk;
import tech.kayys.gollek.sdk.core.exception.SdkException;
import tech.kayys.gollek.sdk.core.model.AsyncJobStatus;
import tech.kayys.gollek.sdk.core.model.BatchInferenceRequest;
import tech.kayys.gollek.sdk.core.model.ModelInfo;
import tech.kayys.gollek.sdk.core.model.PullProgress;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Core interface for the Gollek inference engine SDK.
 * Defines the contract that all implementations must follow.
 */
public interface GollekSdk {
    
    // ==================== Inference Operations ====================
    
    /**
     * Creates a new inference request synchronously.
     *
     * @param request The inference request
     * @return The inference response
     * @throws SdkException if the request fails
     */
    InferenceResponse createCompletion(InferenceRequest request) throws SdkException;
    
    /**
     * Creates a new inference request asynchronously.
     *
     * @param request The inference request
     * @return A CompletableFuture that will complete with the inference response
     */
    java.util.concurrent.CompletableFuture<InferenceResponse> createCompletionAsync(InferenceRequest request);
    
    /**
     * Creates a streaming inference request.
     *
     * @param request The inference request
     * @return A Multi that emits StreamChunk objects
     */
    Multi<StreamChunk> streamCompletion(InferenceRequest request);
    
    // ==================== Job Operations ====================
    
    /**
     * Submits an async inference job.
     *
     * @param request The inference request
     * @return The job ID
     * @throws SdkException if the request fails
     */
    String submitAsyncJob(InferenceRequest request) throws SdkException;
    
    /**
     * Gets the status of an async inference job.
     *
     * @param jobId The job ID
     * @return The job status
     * @throws SdkException if the request fails
     */
    AsyncJobStatus getJobStatus(String jobId) throws SdkException;
    
    /**
     * Waits for an async job to complete.
     *
     * @param jobId The job ID
     * @param maxWaitTime Maximum time to wait
     * @param pollInterval Interval between status checks
     * @return The final job status
     * @throws SdkException if the request fails or times out
     */
    AsyncJobStatus waitForJob(String jobId, java.time.Duration maxWaitTime, java.time.Duration pollInterval) throws SdkException;
    
    /**
     * Performs batch inference.
     *
     * @param batchRequest The batch inference request
     * @return List of inference responses
     * @throws SdkException if the request fails
     */
    List<InferenceResponse> batchInference(BatchInferenceRequest batchRequest) throws SdkException;
    
    // ==================== Provider Operations ====================
    
    /**
     * Lists all available inference providers.
     *
     * @return List of provider information
     * @throws SdkException if the request fails
     */
    List<ProviderInfo> listAvailableProviders() throws SdkException;
    
    /**
     * Gets detailed information about a specific provider.
     *
     * @param providerId The provider ID
     * @return Provider information
     * @throws SdkException if the provider is not found
     */
    ProviderInfo getProviderInfo(String providerId) throws SdkException;
    
    /**
     * Sets the preferred provider for subsequent requests.
     * This can be overridden per-request.
     *
     * @param providerId The provider ID
     * @throws SdkException if the provider is not available
     */
    void setPreferredProvider(String providerId) throws SdkException;
    
    /**
     * Gets the currently preferred provider ID.
     *
     * @return The preferred provider ID, or empty if none is set
     */
    Optional<String> getPreferredProvider();
    
    // ==================== Model Operations ====================
    
    /**
     * Lists all available models.
     *
     * @return List of model information
     * @throws SdkException if the request fails
     */
    List<ModelInfo> listModels() throws SdkException;
    
    /**
     * Lists models with pagination.
     *
     * @param offset Starting offset
     * @param limit Maximum number of models to return
     * @return List of model information
     * @throws SdkException if the request fails
     */
    List<ModelInfo> listModels(int offset, int limit) throws SdkException;
    
    /**
     * Gets detailed information about a specific model.
     *
     * @param modelId The model ID
     * @return Model information, or empty if not found
     * @throws SdkException if the request fails
     */
    Optional<ModelInfo> getModelInfo(String modelId) throws SdkException;
    
    /**
     * Pulls a model from a registry.
     * Supports Ollama models (e.g., "llama2") and HuggingFace models (e.g., "hf:TheBloke/Llama-2").
     *
     * @param modelSpec The model specification
     * @param progressCallback Callback for progress updates (can be null)
     * @throws SdkException if the pull fails
     */
    void pullModel(String modelSpec, Consumer<PullProgress> progressCallback) throws SdkException;
    
    /**
     * Deletes a local model.
     *
     * @param modelId The model ID to delete
     * @throws SdkException if the deletion fails
     */
    void deleteModel(String modelId) throws SdkException;
}
