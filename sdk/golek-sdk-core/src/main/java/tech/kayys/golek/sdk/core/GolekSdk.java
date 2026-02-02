package tech.kayys.golek.sdk.core;

import io.smallrye.mutiny.Multi;
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.stream.StreamChunk;
import tech.kayys.golek.sdk.core.exception.SdkException;
import tech.kayys.golek.sdk.core.model.AsyncJobStatus;
import tech.kayys.golek.sdk.core.model.BatchInferenceRequest;

import java.util.List;

/**
 * Core interface for the Golek inference engine SDK.
 * Defines the contract that all implementations must follow.
 */
public interface GolekSdk {
    
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
}