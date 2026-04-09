package tech.kayys.gollek.engine.inference;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.model.HealthStatus;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.inference.InferenceEngine;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.engine.plugin.PluginSystemIntegrator;

import java.util.Map;
import java.util.UUID;

/**
 * Default implementation of the Gollek Inference Engine.
 * 
 * <p>
 * Integrates with the four-level plugin system:
 * <ul>
 * <li>Level 1: Runner Plugins (model format support)</li>
 * <li>Level 2: Feature Plugins (domain-specific capabilities)</li>
 * <li>Level 3: Optimization Plugins (performance enhancements)</li>
 * <li>Level 4: Kernel Plugins (platform-specific GPU kernels)</li>
 * </ul>
 */
@ApplicationScoped
public class DefaultInferenceEngine implements InferenceEngine {

        private static final Logger LOG = Logger.getLogger(DefaultInferenceEngine.class);

        @Inject
        InferenceOrchestrator orchestrator;

        @Inject
        InferenceMetrics metrics;

        @Inject
        PluginSystemIntegrator pluginIntegrator;

        private boolean initialized = false;
        private boolean healthy = true;
        private long totalInferences = 0;
        private long failedInferences = 0;

        @Override
        public void initialize() {
                LOG.info("Initializing Gollek Inference Engine...");

                // Initialize plugin systems first
                pluginIntegrator.initialize();

                // Then initialize orchestrator
                orchestrator.initialize();

                initialized = true;
                healthy = true;

                // Log initial metrics configuration
                LOG.info("LLM Metrics initialized with TTFT, TPOT, ITL, and throughput tracking");

                // Log plugin system status
                Map<String, Boolean> pluginStatus = pluginIntegrator.getPluginStatus();
                LOG.info("Plugin System Status:");
                pluginStatus.forEach(
                                (level, status) -> LOG.infof("  %s: %s", level, status ? "✓ Enabled" : "✗ Disabled"));
        }

        @Override
        public void shutdown() {
                LOG.info("Shutting down Gollek Inference Engine...");

                // Shutdown orchestrator first
                orchestrator.shutdown();

                // Then shutdown plugin systems
                pluginIntegrator.shutdown();

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

                return orchestrator.streamExecute(request.getModel(), request);
        }

        @Override
        public Uni<InferenceResponse> executeAsync(String modelId, InferenceRequest request) {
                return orchestrator.executeAsync(modelId, request);
        }

        @Override
        public InferenceResponse execute(String modelId, InferenceRequest request) {
                return orchestrator.execute(modelId, request)
                                .await().atMost(java.time.Duration.ofSeconds(60));
        }

        @Override
        public Multi<StreamingInferenceChunk> streamExecute(String modelId, InferenceRequest request) {
                return orchestrator.streamExecute(modelId, request);
        }

        @Override
        public Uni<tech.kayys.gollek.spi.embedding.EmbeddingResponse> executeEmbedding(String modelId,
                        tech.kayys.gollek.spi.embedding.EmbeddingRequest request) {
                return orchestrator.executeEmbedding(modelId, request);
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
