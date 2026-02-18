package tech.kayys.gollek.engine.inference;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import io.opentelemetry.api.trace.Span;

// API imports
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.InferenceStage;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.stream.StreamChunk;
import tech.kayys.gollek.engine.observability.MetricsPublisher;
import tech.kayys.gollek.engine.routing.ModelRouterService;

/**
 * Orchestrates inference requests across multiple runners with
 * intelligent routing, fallback, load balancing, and stage-aware
 * disaggregated prefill/decode support.
 */
@ApplicationScoped
public class DefaultInferenceOrchestrator implements InferenceOrchestrator {

        private static final Logger LOG = Logger.getLogger(InferenceOrchestrator.class);

        /** Default token threshold below which disaggregation is skipped. */
        static final int DEFAULT_SMALL_PROMPT_THRESHOLD = 128;

        private final ModelRouterService router;
        private final MetricsPublisher metrics;

        // Configuration (can be injected via @ConfigProperty in a future phase)
        private volatile boolean disaggregatedMode = false;
        private volatile int smallPromptThreshold = DEFAULT_SMALL_PROMPT_THRESHOLD;

        @Inject
        public DefaultInferenceOrchestrator(
                        ModelRouterService router,
                        MetricsPublisher metrics) {
                this.router = router;
                this.metrics = metrics;
        }

        /**
         * Execute inference with automatic runner selection, stage-aware routing,
         * and fallback (async).
         */
        public Uni<InferenceResponse> executeAsync(
                        String modelId,
                        InferenceRequest request) {
                var span = Span.current();
                span.setAttribute("model.id", modelId);
                String tenantId = request.getMetadata().getOrDefault("tenantId", "community").toString();
                span.setAttribute("tenant.id", tenantId);

                // — Stage-aware routing —
                InferenceRequest stagedRequest = resolveStage(request);
                InferenceStage stage = stagedRequest.getInferenceStage();
                span.setAttribute("inference.stage", stage.name());

                LOG.infof("Orchestrating inference for model %s (tenant: %s, stage: %s)",
                                modelId, tenantId, stage.getDisplayName());

                return routeForStage(modelId, stagedRequest, stage)
                                .onItem().invoke(response -> {
                                        // Record success metrics
                                        metrics.recordSuccess("unified", modelId, response.getDurationMs());

                                        LOG.infof("Inference completed: request=%s, duration=%dms, tokens=%d",
                                                        request.getRequestId(), response.getDurationMs(),
                                                        response.getTokensUsed());
                                })
                                .onFailure().invoke(error -> {
                                        // Record failure metrics
                                        metrics.recordFailure("unified", modelId, error.getClass().getSimpleName());

                                        LOG.errorf(error, "Inference orchestration failed for model %s (stage: %s)",
                                                        modelId, stage.getDisplayName());
                                });
        }

        /**
         * Execute inference with automatic runner selection and fallback (sync)
         */
        public InferenceResponse execute(
                        String modelId,
                        InferenceRequest request) {
                LOG.debugf("Starting synchronous inference for model: %s", modelId);

                return executeAsync(modelId, request)
                                .await().indefinitely();
        }

        /**
         * Execute streaming inference with automatic runner selection
         */
        public Multi<StreamChunk> streamExecute(
                        String modelId,
                        InferenceRequest request) {
                InferenceRequest stagedRequest = resolveStage(request);
                LOG.infof("Streaming inference orchestration for model %s (stage: %s)",
                                modelId, stagedRequest.getInferenceStage().getDisplayName());

                return router.routeStream(modelId, stagedRequest);
        }

        // ---- Stage-aware internals ----

        /**
         * Resolve the inference stage for a request if not already set.
         * Uses prompt token count and disaggregation config.
         */
        InferenceRequest resolveStage(InferenceRequest request) {
                if (request.hasExplicitStage()) {
                        return request; // Already stamped
                }

                int tokenCount = request.getPromptTokenCount();
                if (tokenCount < 0) {
                        // Estimate from message count (rough heuristic: ~4 tokens per char / 5 chars
                        // per word)
                        tokenCount = estimateTokenCount(request);
                }

                InferenceStage stage = InferenceStage.forRequest(
                                tokenCount, disaggregatedMode, smallPromptThreshold);

                return request.toBuilder()
                                .inferenceStage(stage)
                                .promptTokenCount(tokenCount)
                                .build();
        }

        /**
         * Route to the appropriate provider based on the resolved stage.
         * <p>
         * In disaggregated mode with PREFILL stage, this will eventually route
         * to a prefill-capable node. For now, all stages route through the
         * standard router (Phase 4 will add gRPC handoff).
         */
        private Uni<InferenceResponse> routeForStage(
                        String modelId,
                        InferenceRequest request,
                        InferenceStage stage) {

                if (stage.isDisaggregated()) {
                        LOG.debugf("Disaggregated routing for stage %s (model: %s, tokens: %d)",
                                        stage.getDisplayName(), modelId, request.getPromptTokenCount());
                        // Phase 4: Route PREFILL to prefill node, DECODE to decode node
                        // For now, fall through to unified routing
                }

                return router.route(modelId, request);
        }

        /**
         * Rough token count estimate from message text.
         * A proper tokenizer will replace this in production.
         */
        static int estimateTokenCount(InferenceRequest request) {
                int charCount = 0;
                for (var msg : request.getMessages()) {
                        String content = msg.getContent();
                        if (content != null) {
                                charCount += content.length();
                        }
                }
                // Rough approximation: ~4 chars per token for English text
                return Math.max(1, charCount / 4);
        }

        // ---- Configuration ----

        /**
         * Enable or disable disaggregated prefill/decode mode.
         */
        public void setDisaggregatedMode(boolean enabled) {
                this.disaggregatedMode = enabled;
                LOG.infof("Disaggregated mode %s", enabled ? "ENABLED" : "DISABLED");
        }

        /**
         * Check if disaggregated mode is active.
         */
        public boolean isDisaggregatedMode() {
                return disaggregatedMode;
        }

        /**
         * Set the prompt token threshold below which disaggregation is skipped.
         */
        public void setSmallPromptThreshold(int threshold) {
                this.smallPromptThreshold = threshold;
        }
}
