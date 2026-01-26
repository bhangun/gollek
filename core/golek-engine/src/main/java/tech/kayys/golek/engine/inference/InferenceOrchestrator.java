package tech.kayys.golek.engine.inference;

import java.util.List;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributeKey;
import java.time.Duration;

// API imports
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.tenant.TenantContext;

// Exception imports
import tech.kayys.golek.core.exception.InferenceException;
import tech.kayys.golek.core.exception.ModelNotFoundException;
import tech.kayys.golek.core.exception.TenantQuotaExceededException;
import tech.kayys.golek.core.exception.ValidationException;
import tech.kayys.golek.core.exception.AllRunnersFailedException;

// Model imports
import tech.kayys.golek.model.ModelManifest;
import tech.kayys.golek.engine.model.RequestContext;
import tech.kayys.golek.engine.model.RunnerCandidate;
import tech.kayys.golek.engine.model.ModelRepository;

// Execution and infrastructure imports
import tech.kayys.golek.api.model.ModelRunner;
import tech.kayys.golek.model.core.ModelRunnerFactory;
import tech.kayys.golek.engine.observability.MetricsPublisher;
import tech.kayys.golek.engine.resource.CircuitBreaker;

/**
 * Orchestrates inference requests across multiple runners with
 * intelligent routing, fallback, and load balancing.
 */
@ApplicationScoped
public class InferenceOrchestrator {

        private static final Logger LOG = Logger.getLogger(InferenceOrchestrator.class);

        private final ModelRouterService router;
        private final ModelRunnerFactory factory;
        private final ModelRepository repository;
        private final MetricsPublisher metrics;
        private final CircuitBreaker circuitBreaker;

        @Inject
        public InferenceOrchestrator(
                        ModelRouterService router,
                        ModelRunnerFactory factory,
                        ModelRepository repository,
                        MetricsPublisher metrics,
                        CircuitBreaker circuitBreaker) {
                this.router = router;
                this.factory = factory;
                this.repository = repository;
                this.metrics = metrics;
                this.circuitBreaker = circuitBreaker;
        }

        /**
         * Execute inference with automatic runner selection and fallback (async)
         */
        public io.smallrye.mutiny.Uni<InferenceResponse> executeAsync(
                        String modelId,
                        InferenceRequest request,
                        TenantContext tenantContext) {
                var span = Span.current();
                span.setAttribute("model.id", modelId);
                span.setAttribute("tenant.id", tenantContext.tenantId().value());

                // Load model manifest
                ModelManifest manifest = repository
                                .findById(modelId, tenantContext.tenantId())
                                .orElseThrow(() -> new ModelNotFoundException(modelId));

                // Build request context with timeout and priority
                RequestContext ctx = RequestContext.builder()
                                .tenantContext(tenantContext)
                                .timeout(Duration.ofSeconds(30))
                                .priority(request.priority())
                                .preferredDevice(request.deviceHint())
                                .build();

                // Select and rank candidate runners
                List<RunnerCandidate> candidates = router.selectRunners(
                                manifest,
                                ctx);

                // Use Uni to handle the fallback logic asynchronously
                return executeWithFallback(candidates, manifest, request, ctx, modelId, 0, null);
        }

        private io.smallrye.mutiny.Uni<InferenceResponse> executeWithFallback(
                        List<RunnerCandidate> candidates,
                        ModelManifest manifest,
                        InferenceRequest request,
                        RequestContext ctx,
                        String modelId,
                        int currentIndex,
                        Throwable lastError) {

                if (currentIndex >= candidates.size()) {
                        LOG.errorf("All runners failed for model %s", modelId);
                        return io.smallrye.mutiny.Uni.createFrom().failure(
                                        new AllRunnersFailedException(
                                                        "All runners failed for model " + modelId,
                                                        lastError));
                }

                RunnerCandidate candidate = candidates.get(currentIndex);

                LOG.debugf("Attempting inference with runner: %s for model: %s",
                                candidate.runnerName(), modelId);

                return executeWithRunner(manifest, candidate, request, ctx)
                                .onItem().transform(response -> {
                                        LOG.infof("Inference successful with runner: %s for model: %s",
                                                        candidate.runnerName(), modelId);
                                        return response;
                                })
                                .onFailure().recoverWithUni(error -> {
                                        LOG.warnf(error, "Runner %s failed for model %s, attempting fallback",
                                                        candidate.runnerName(), modelId);

                                        metrics.recordFailure(
                                                        candidate.runnerName(),
                                                        modelId,
                                                        error.getClass().getSimpleName());

                                        var span = Span.current();
                                        span.addEvent("Runner failed, attempting fallback",
                                                        Attributes.of(
                                                                        AttributeKey.stringKey("runner"),
                                                                        candidate.runnerName(),
                                                                        AttributeKey.stringKey("error"), error.getMessage()));

                                        // Don't retry on quota or validation errors
                                        if (error instanceof TenantQuotaExceededException ||
                                                        error instanceof ValidationException) {
                                                LOG.debugf("Not retrying due to non-retryable error: %s",
                                                                error.getClass().getSimpleName());
                                                return io.smallrye.mutiny.Uni.createFrom().failure(error);
                                        }

                                        // Try next candidate
                                        return executeWithFallback(
                                                        candidates,
                                                        manifest,
                                                        request,
                                                        ctx,
                                                        modelId,
                                                        currentIndex + 1,
                                                        error);
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

        private io.smallrye.mutiny.Uni<InferenceResponse> executeWithRunner(
                        ModelManifest manifest,
                        RunnerCandidate candidate,
                        InferenceRequest request,
                        RequestContext ctx) {
                var timer = metrics.startTimer();

                // Get or create runner instance
                ModelRunner runner = factory.getRunner(
                                manifest,
                                candidate.runnerName(),
                                ctx.tenantContext());

                // Execute with circuit breaker protection
                return circuitBreaker.call(() -> runner.infer(request, ctx))
                                .onItem().transform(response -> {
                                        metrics.recordSuccess(
                                                        candidate.runnerName(),
                                                        manifest.modelId(),
                                                        timer.stop());
                                        return response;
                                })
                                .onFailure().recoverWithUni(error -> {
                                        metrics.recordFailure(
                                                        candidate.runnerName(),
                                                        manifest.modelId(),
                                                        error.getClass().getSimpleName());

                                        LOG.errorf(error, "Inference failed with runner: %s for model: %s",
                                                        candidate.runnerName(), manifest.modelId());

                                        return io.smallrye.mutiny.Uni.createFrom().failure(
                                                        new InferenceException(
                                                                        "Inference failed with runner: " + candidate.runnerName(),
                                                                        error));
                                });
        }
}