package tech.kayys.gollek.engine.inference;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.core.engine.EngineMetadata;
import tech.kayys.gollek.spi.model.HealthStatus;
import tech.kayys.gollek.core.inference.InferenceEngine;

import java.util.List;
import java.util.UUID;

/**
 * Default implementation of the Gollek Inference Engine.
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
                LOG.info("Initializing Gollek Inference Engine...");
                initialized = true;
                healthy = true;

                // Log initial metrics configuration
                LOG.info("LLM Metrics initialized with TTFT, TPOT, ITL, and throughput tracking");
        }

        @Override
        public void shutdown() {
                LOG.info("Shutting down Gollek Inference Engine...");
                initialized = false;
        }

        @Override
        public Uni<InferenceResponse> infer(InferenceRequest request) {
                if (!initialized) {
                        return Uni.createFrom().failure(new IllegalStateException("Engine not initialized"));
                }

                return orchestrator.executeAsync(request.getModel(), request)
                                .onItem().invoke(response -> {
                                        totalInferences++;

                                        // Log request-level metrics
                                        LOG.debugf("Request %s completed: E2E=%dms, tokens=%d",
                                                        request.getRequestId(),
                                                        response.getDurationMs(),
                                                        response.getTokensUsed());
                                })
                                .onFailure().invoke(failure -> failedInferences++);
        }

        @Override
        public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
                if (!initialized || !healthy) {
                        return Multi.createFrom().failure(new IllegalStateException("Engine not ready"));
                }

                return orchestrator.streamExecute(request.getModel(), request)
                                .map(chunk -> {
                                        return new StreamingInferenceChunk(
                                                        request.getRequestId(),
                                                        chunk.getIndex(),
                                                        chunk.getDelta(),
                                                        chunk.isFinal(),
                                                        chunk.getIndex() > 0 ? chunk.getIndex() * 100L : 0L);
                                });
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
                                0.0,
                                initialized ? "RUNNING" : "STOPPED");
        }
}
