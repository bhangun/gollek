package tech.kayys.golek.runtime;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.inference.StreamChunk;

/**
 * Core inference engine interface that manages model execution
 */
public interface InferenceEngine {
    
    /**
     * Initialize the inference engine
     */
    void initialize();
    
    /**
     * Execute synchronous inference
     */
    Uni<InferenceResponse> infer(InferenceRequest request);
    
    /**
     * Execute streaming inference
     */
    Multi<StreamChunk> stream(InferenceRequest request);
    
    /**
     * Shutdown the inference engine gracefully
     */
    void shutdown();
    
    /**
     * Get engine health status
     */
    boolean isHealthy();
    
    /**
     * Get engine statistics
     */
    EngineStats getStats();
    
    /**
     * Engine statistics data
     */
    record EngineStats(
        long activeInferences,
        long totalInferences,
        long failedInferences,
        double avgLatencyMs,
        String status
    ) {}
}