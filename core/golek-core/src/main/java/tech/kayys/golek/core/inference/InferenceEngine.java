package tech.kayys.golek.core.inference;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.core.engine.EngineMetadata;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.spi.model.HealthStatus;

/**
 * Main entry point for inference requests.
 * Thread-safe and stateless.
 */
public interface InferenceEngine {

        /**
         * Execute synchronous inference
         */
        Uni<InferenceResponse> infer(
                        InferenceRequest request,
                        TenantContext tenantContext);

        /**
         * Execute streaming inference
         */
        Uni<StreamingResponse> inferStream(
                        InferenceRequest request,
                        TenantContext tenantContext);

        /**
         * Get engine metadata
         */
        EngineMetadata metadata();

        /**
         * Health check
         */
        HealthStatus health();
}