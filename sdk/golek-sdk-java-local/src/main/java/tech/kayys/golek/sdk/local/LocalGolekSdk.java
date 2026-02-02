package tech.kayys.golek.sdk.local;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.stream.StreamChunk;
import tech.kayys.golek.engine.inference.InferenceService;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.exception.SdkException;
import tech.kayys.golek.sdk.core.model.AsyncJobStatus;
import tech.kayys.golek.sdk.core.model.BatchInferenceRequest;

import java.time.Instant;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Local implementation of the Golek SDK that runs within the same JVM as the inference engine.
 * This implementation directly calls the internal services without HTTP overhead.
 */
@ApplicationScoped
public class LocalGolekSdk implements GolekSdk {
    
    @Inject
    InferenceService inferenceService;
    
    /**
     * Creates a new inference request synchronously.
     *
     * @param request The inference request
     * @return The inference response
     * @throws SdkException if the request fails
     */
    @Override
    public InferenceResponse createCompletion(InferenceRequest request) throws SdkException {
        try {
            // Call the internal inference service directly
            return inferenceService.inferAsync(request).await().indefinitely();
        } catch (Exception e) {
            throw new SdkException("Failed to create completion", e);
        }
    }
    
    /**
     * Creates a new inference request asynchronously.
     *
     * @param request The inference request
     * @return A CompletableFuture that will complete with the inference response
     */
    @Override
    public CompletableFuture<InferenceResponse> createCompletionAsync(InferenceRequest request) {
        return inferenceService.inferAsync(request)
            .subscribeAsCompletionStage();
    }
    
    /**
     * Creates a streaming inference request.
     *
     * @param request The inference request
     * @return A Multi that emits StreamChunk objects
     */
    @Override
    public Multi<StreamChunk> streamCompletion(InferenceRequest request) {
        try {
            // Call the internal streaming service directly
            return inferenceService.inferStream(request)
                .map(chunk -> new StreamChunk(
                    chunk.index(),
                    chunk.delta(),
                    chunk.isFinal()
                ));
        } catch (Exception e) {
            return Multi.createFrom().failure(new SdkException("Failed to initiate streaming completion", e));
        }
    }
    
    /**
     * Submits an async inference job.
     *
     * @param request The inference request
     * @return The job ID
     * @throws SdkException if the request fails
     */
    @Override
    public String submitAsyncJob(InferenceRequest request) throws SdkException {
        try {
            // Submit the job to the internal async job manager
            return inferenceService.submitAsyncJob(request).await().indefinitely();
        } catch (Exception e) {
            throw new SdkException("Failed to submit async job", e);
        }
    }
    
    /**
     * Gets the status of an async inference job.
     *
     * @param jobId The job ID
     * @return The job status
     * @throws SdkException if the request fails
     */
    @Override
    public AsyncJobStatus getJobStatus(String jobId) throws SdkException {
        try {
            // Get the job status from the internal service
            // For now, using a default tenant ID - in a real implementation,
            // this would come from context or be passed as a parameter
            var status = inferenceService.getJobStatus(jobId, "default").await().indefinitely();

            return new AsyncJobStatus(
                status.jobId(),
                status.requestId(),
                status.tenantId(),
                status.status(),
                status.result(),
                status.error(),
                status.submittedAt(),
                status.completedAt()
            );
        } catch (Exception e) {
            throw new SdkException("Failed to get job status", e);
        }
    }
    
    /**
     * Waits for an async job to complete.
     *
     * @param jobId The job ID
     * @param maxWaitTime Maximum time to wait
     * @param pollInterval Interval between status checks
     * @return The final job status
     * @throws SdkException if the request fails or times out
     */
    @Override
    public AsyncJobStatus waitForJob(String jobId, Duration maxWaitTime, Duration pollInterval) throws SdkException {
        long startTime = System.currentTimeMillis();
        long maxWaitMillis = maxWaitTime.toMillis();
        
        while (System.currentTimeMillis() - startTime < maxWaitMillis) {
            AsyncJobStatus status = getJobStatus(jobId);
            
            if (status.isComplete()) {
                return status;
            }
            
            try {
                Thread.sleep(pollInterval.toMillis());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new SdkException("Job polling interrupted", e);
            }
        }
        
        throw new SdkException("Job " + jobId + " did not complete within the specified time");
    }
    
    /**
     * Performs batch inference.
     *
     * @param batchRequest The batch inference request
     * @return List of inference responses
     * @throws SdkException if the request fails
     */
    @Override
    public List<InferenceResponse> batchInference(BatchInferenceRequest batchRequest) throws SdkException {
        try {
            // Convert the core BatchInferenceRequest to the internal representation
            var internalBatchRequest = new tech.kayys.golek.engine.inference.InferenceService.BatchInferenceRequest(
                batchRequest.getRequests(),
                batchRequest.getMaxConcurrent() != null ? batchRequest.getMaxConcurrent() : 5
            );

            // Call the internal batch inference method
            return inferenceService.batchInfer(internalBatchRequest, "default").await().indefinitely();
        } catch (Exception e) {
            throw new SdkException("Failed to perform batch inference", e);
        }
    }
}