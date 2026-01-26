package tech.kayys.golek.api.model;

import java.util.List;
import java.util.concurrent.CompletionStage;

import tech.kayys.golek.api.context.RequestContext;
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;

/**
 * Core SPI for model execution backends.
 * All model runner implementations must implement this interface.
 * 
 * <p>This is the canonical interface for model runners - all other
 * versions are deprecated and will be removed in future versions.</p>
 */
public interface ModelRunner extends AutoCloseable {

    /**
     * Execute synchronous inference
     * 
     * @param request Inference request with inputs
     * @param context Request context with timeout, priority, etc.
     * @return Inference response with outputs
     * @throws tech.kayys.golek.api.exception.InferenceException if execution fails
     */
    InferenceResponse infer(
            InferenceRequest request,
            RequestContext context);

    /**
     * Execute asynchronous inference with callback
     * 
     * @param request Inference request
     * @param context Request context
     * @return CompletionStage for async processing
     */
    CompletionStage<InferenceResponse> inferAsync(
            InferenceRequest request,
            RequestContext context);

    /**
     * Health check for this runner instance
     * 
     * @return Health status with diagnostics
     */
    HealthStatus health();

    /**
     * Get current resource utilization metrics
     * 
     * @return Resource usage snapshot
     */
    ResourceMetrics getMetrics();

    /**
     * Warm up the model (optional optimization)
     * 
     * @param sampleInputs Sample inputs for warming
     */
    default void warmup(List<InferenceRequest> sampleInputs) {
        // Default no-op
    }

    /**
     * Get runner metadata
     * 
     * @return Metadata about this runner implementation
     */
    RunnerMetadata metadata();

    /**
     * Gracefully release resources
     */
    @Override
    void close();
}
