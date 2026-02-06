package tech.kayys.golek.engine.inference;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import io.opentelemetry.api.trace.Span;

// API imports
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import io.smallrye.mutiny.Multi;
import tech.kayys.golek.spi.stream.StreamChunk;
import tech.kayys.golek.engine.observability.MetricsPublisher;
import tech.kayys.golek.engine.routing.ModelRouterService;
import tech.kayys.wayang.tenant.TenantContext;

/**
 * Orchestrates inference requests across multiple runners with
 * intelligent routing, fallback, and load balancing.
 */
@ApplicationScoped
public class InferenceOrchestrator {

        private static final Logger LOG = Logger.getLogger(InferenceOrchestrator.class);

        private final ModelRouterService router;
        private final MetricsPublisher metrics;

        @Inject
        public InferenceOrchestrator(
                        ModelRouterService router,
                        MetricsPublisher metrics) {
                this.router = router;
                this.metrics = metrics;
        }

        /**
         * Execute inference with automatic runner selection and fallback (async)
         */
    public io.smallrye.mutiny.Uni<InferenceResponse> executeAsync(
            String modelId,
            InferenceRequest request,
            TenantContext tenantContext) {
        TenantContext effectiveTenantContext = ensureTenantContext(tenantContext);
        var span = Span.current();
        span.setAttribute("model.id", modelId);
        span.setAttribute("tenant.id", effectiveTenantContext.getTenantId().value());

        LOG.infof("Orchestrating inference for model %s (tenant: %s)",
                        modelId, effectiveTenantContext.getTenantId().value());

        return router.route(modelId, request, effectiveTenantContext)
                        .onItem().invoke(response -> {
                                metrics.recordSuccess("unified", modelId, 0); // Latency recorded in router
                        })
                                .onFailure().invoke(error -> {
                                        metrics.recordFailure("unified", modelId, error.getClass().getSimpleName());
                                        LOG.errorf(error, "Inference orchestration failed for model %s", modelId);
                                });
        }

        /**
         * Execute inference with automatic runner selection and fallback (sync)
         */
    public InferenceResponse execute(
            String modelId,
            InferenceRequest request,
            TenantContext tenantContext) {
        LOG.debugf("Starting synchronous inference for model: %s", modelId);

        return executeAsync(modelId, request, tenantContext)
                        .await().indefinitely(); // Note: Use with caution in reactive contexts
    }

        /**
         * Execute streaming inference with automatic runner selection
         */
    public Multi<StreamChunk> streamExecute(
            String modelId,
            InferenceRequest request,
            TenantContext tenantContext) {
        TenantContext effectiveTenantContext = ensureTenantContext(tenantContext);
        LOG.infof("Streaming inference orchestration for model %s", modelId);
        return router.routeStream(modelId, request, effectiveTenantContext);
    }

    private TenantContext ensureTenantContext(TenantContext tenantContext) {
        return tenantContext != null ? tenantContext : TenantContext.of("default");
    }
}
