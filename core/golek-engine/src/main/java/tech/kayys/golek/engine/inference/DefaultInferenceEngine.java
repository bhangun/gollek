package tech.kayys.golek.engine.inference;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.golek.spi.inference.StreamingInferenceChunk;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.core.engine.EngineMetadata;
import tech.kayys.golek.spi.model.HealthStatus;
import tech.kayys.golek.core.inference.InferenceEngine;
import tech.kayys.wayang.tenant.TenantContext;

import java.util.List;
import java.util.UUID;

/**
 * Default implementation of the Golek Inference Engine.
 */
@ApplicationScoped
public class DefaultInferenceEngine implements InferenceEngine {

        private static final Logger LOG = Logger.getLogger(DefaultInferenceEngine.class);

        @Inject
        InferenceOrchestrator orchestrator;

        @Inject
        InferenceMetrics metrics;

        private boolean initialized = false;
        private boolean healthy = true;
        private long totalInferences = 0;
        private long failedInferences = 0;

        @Override
        public void initialize() {
                LOG.info("Initializing Golek Inference Engine...");
                initialized = true;
                healthy = true;
        }

        @Override
        public void shutdown() {
                LOG.info("Shutting down Golek Inference Engine...");
                initialized = false;
        }

        @Override
        public Uni<InferenceResponse> infer(InferenceRequest request, TenantContext tenantContext) {
                if (!initialized) {
                        return Uni.createFrom().failure(new IllegalStateException("Engine not initialized"));
                }

                return orchestrator.executeAsync(request.getModel(), request, tenantContext)
                                .onItem().invoke(response -> totalInferences++)
                                .onFailure().invoke(failure -> failedInferences++);
        }

        @Override
        public Multi<StreamingInferenceChunk> stream(InferenceRequest request, TenantContext tenantContext) {
                if (!initialized || !healthy) {
                        return Multi.createFrom().failure(new IllegalStateException("Engine not ready"));
                }

                TenantContext effectiveTenant = tenantContext != null ? tenantContext : TenantContext.of("community");
                return orchestrator.streamExecute(request.getModel(), request, effectiveTenant)
                                .map(chunk -> new StreamingInferenceChunk(
                                                request.getRequestId(),
                                                0, // sequence not easily available from StreamChunk
                                                chunk.getDelta(),
                                                chunk.isFinal(),
                                                0L));
        }

        @Override
        public Uni<String> submitAsyncJob(InferenceRequest request) {
                return Uni.createFrom().item(() -> {
                        String jobId = UUID.randomUUID().toString();
                        LOG.infof("Submitted async job %s for model %s", jobId, request.getModel());
                        return jobId;
                });
        }

        @Override
        public EngineMetadata metadata() {
                return EngineMetadata.builder()
                                .version("1.0.0")
                                .supportedPhases(List.of("unified"))
                                .build();
        }

        @Override
        public HealthStatus health() {
                return healthy ? HealthStatus.healthy("Engine is operational")
                                : HealthStatus.unhealthy("Engine is not operational");
        }

        @Override
        public boolean isHealthy() {
                return healthy && initialized;
        }

        @Override
        public EngineStats getStats() {
                return new EngineStats(
                                0,
                                totalInferences,
                                failedInferences,
                                0,
                                initialized ? "RUNNING" : "STOPPED");
        }
}
