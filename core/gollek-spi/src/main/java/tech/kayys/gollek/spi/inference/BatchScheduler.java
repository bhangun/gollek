package tech.kayys.gollek.spi.inference;

import java.util.concurrent.CompletableFuture;

/**
 * Manages the batching of inference requests.
 * Implementations will handle logic for STATIC, DYNAMIC, or CONTINUOUS batch
 * scheduling.
 */
public interface BatchScheduler {

    /**
     * Submits an inference request to the scheduling queue.
     * 
     * @param request the inference request to enqueue
     * @return a CompletableFuture indicating completion of the inference
     */
    CompletableFuture<InferenceResponse> submit(InferenceRequest request);

    /**
     * Submits a pre-formed batch for execution.
     * 
     * @param batch the batch request
     * @return a CompletableFuture containing the batch response/results
     */
    CompletableFuture<BatchResponse> submitBatch(BatchInferenceRequest batch);

    /**
     * Forces all queued requests to be dispatched immediately, bypassing wait
     * times.
     */
    void flush();

    /**
     * Get the current batching configuration.
     * 
     * @return the configuration
     */
    BatchConfig getConfig();

    /**
     * Dynamically updates the batching strategy and configuration at runtime.
     * 
     * @param config the new configuration
     */
    void setConfig(BatchConfig config);
}
