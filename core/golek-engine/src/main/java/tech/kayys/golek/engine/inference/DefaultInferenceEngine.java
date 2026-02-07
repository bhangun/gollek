package tech.kayys.golek.engine.inference;

import java.time.Duration;
import java.time.Instant;

import io.smallrye.mutiny.Uni;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.validation.ValidationException;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import tech.kayys.golek.core.inference.StreamingResponse;
import tech.kayys.golek.spi.inference.InferencePhase;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.core.engine.EngineMetadata;
import tech.kayys.golek.spi.model.HealthStatus;
import tech.kayys.golek.core.exception.AuthorizationException;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.spi.exception.InferenceException;
import tech.kayys.golek.core.exception.TenantQuotaExceededException;
import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.engine.execution.ExecutionSignal;
import tech.kayys.golek.engine.execution.ExecutionStateMachine;
import tech.kayys.golek.spi.execution.ExecutionStatus;
import tech.kayys.golek.core.execution.ExecutionToken;
import tech.kayys.golek.core.pipeline.InferencePipeline;
import tech.kayys.golek.engine.execution.DefaultExecutionContext;
import tech.kayys.golek.engine.observability.InferenceMetricsCollector;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.core.inference.InferenceEngine;

/**
 * State-driven inference engine with deterministic execution lifecycle.
 */
@ApplicationScoped
public class DefaultInferenceEngine implements InferenceEngine {

        private static final Logger LOG = Logger.getLogger(DefaultInferenceEngine.class);

        @Inject
        InferencePipeline pipeline;

        @Inject
        ExecutionStateMachine stateMachine;

        @Inject
        EngineContext engineContext;

        @Inject
        InferenceMetricsCollector metrics;

        @ConfigProperty(name = "wayang.inference.retry.max-attempts", defaultValue = "3")
        int maxRetryAttempts;

        @ConfigProperty(name = "wayang.inference.retry.initial-backoff", defaultValue = "1s")
        Duration initialBackoff;

        @ConfigProperty(name = "wayang.inference.retry.max-backoff", defaultValue = "60s")
        Duration maxBackoff;

        @ConfigProperty(name = "wayang.inference.sync.timeout", defaultValue = "5m")
        Duration syncTimeout;

        @Override
        public Uni<InferenceResponse> infer(
                        InferenceRequest request,
                        TenantContext tenantContext) {
                LOG.debugf("Starting inference request: %s", request.getRequestId());

                return createExecutionToken(request, tenantContext)
                                .onItem().transformToUni(token -> executeWithRetry(token, request, tenantContext))
                                .onItem().invoke(response -> LOG.debugf("Completed inference request: %s",
                                                request.getRequestId()));
        }

        private Uni<ExecutionToken> createExecutionToken(InferenceRequest request, TenantContext tenantContext) {
                return Uni.createFrom().item(ExecutionToken.builder()
                                .requestId(request.getRequestId())
                                .variable("tenantId", tenantContext.getTenantId().value())
                                .status(ExecutionStatus.CREATED)
                                .currentPhase(InferencePhase.PRE_VALIDATE)
                                .build());
        }

        private Uni<InferenceResponse> executeWithRetry(ExecutionToken token, InferenceRequest request,
                        TenantContext tenantContext) {
                Instant startTime = Instant.now();
                // Create execution context
                ExecutionContext execContext = new DefaultExecutionContext(
                                engineContext,
                                tenantContext,
                                token);

                // Transition to RUNNING
                ExecutionStatus nextStatus = stateMachine.next(
                                token.getStatus(),
                                ExecutionSignal.START);
                execContext.updateStatus(nextStatus);

                // Execute pipeline with metrics
                return executeWithStateMachine(execContext)
                                .onItem().transformToUni(ctx -> {
                                        // Final status check
                                        if (ctx.token().getStatus() == ExecutionStatus.COMPLETED) {
                                                LOG.infof("Inference complete: %s", ctx.token().getRequestId());
                                        }
                                        return Uni.createFrom().item(ctx.token());
                                })
                                .onItem().transform(tokenResult -> {
                                        // Extract response from context
                                        InferenceResponse response = execContext
                                                        .getVariable("response", InferenceResponse.class)
                                                        .orElseThrow(() -> new InferenceException(
                                                                        "No response generated"));

                                        // Record success metrics
                                        Duration duration = Duration.between(startTime, Instant.now());
                                        metrics.recordSuccess(duration);
                                        LOG.infof("Inference request %s completed successfully in %d ms",
                                                        request.getRequestId(), duration.toMillis());

                                        return response;
                                })
                                .onFailure().invoke(error -> {
                                        // Record failure metrics
                                        Duration duration = Duration.between(startTime, Instant.now());
                                        metrics.recordFailure(error.getClass().getSimpleName(), duration);
                                        LOG.errorf(error, "Inference request %s failed after %d ms",
                                                        request.getRequestId(), duration.toMillis());
                                });
        }

        private Uni<ExecutionContext> executeWithStateMachine(
                        ExecutionContext context) {
                return pipeline.execute(context)
                                .onItem().invoke(ctx -> {
                                        // Transition to COMPLETED on success
                                        ExecutionStatus nextStatus = stateMachine.next(
                                                        ctx.token().getStatus(),
                                                        ExecutionSignal.EXECUTION_SUCCESS);
                                        ctx.updateStatus(nextStatus);
                                })
                                .onFailure().recoverWithUni(error -> {
                                        // Handle failure with retry logic
                                        return handleFailure(context, error);
                                });
        }

        private Uni<ExecutionContext> handleFailure(
                        ExecutionContext context,
                        Throwable error) {
                int attempt = context.token().getAttempt();

                if (attempt < maxRetryAttempts && isRetryable(error)) {
                        // Record retry attempt
                        metrics.recordRetry(attempt + 1);

                        LOG.warnf("Inference request %s failed (attempt %d/%d), retrying: %s",
                                        context.token().getRequestId(), attempt + 1, maxRetryAttempts,
                                        error.getMessage());

                        // Transition to RETRYING
                        ExecutionStatus nextStatus = stateMachine.next(
                                        context.token().getStatus(),
                                        ExecutionSignal.PHASE_FAILURE);
                        context.updateStatus(nextStatus);
                        context.incrementAttempt();

                        // Calculate backoff with configured values
                        Duration backoff = calculateBackoff(attempt);
                        LOG.debugf("Waiting %d ms before retry", backoff.toMillis());

                        return Uni.createFrom().item(context)
                                        .onItem().delayIt().by(backoff)
                                        .chain(() -> {
                                                // Transition back to RUNNING
                                                ExecutionStatus resumed = stateMachine.next(
                                                                context.token().getStatus(),
                                                                ExecutionSignal.START);
                                                context.updateStatus(resumed);
                                                return executeWithStateMachine(context);
                                        });
                } else {
                        LOG.errorf(error, "Inference request %s failed after %d attempts",
                                        context.token().getRequestId(), attempt);

                        // Exhausted retries, transition to FAILED
                        ExecutionStatus failedStatus = stateMachine.next(
                                        context.token().getStatus(),
                                        ExecutionSignal.RETRY_EXHAUSTED);
                        context.updateStatus(failedStatus);

                        return Uni.createFrom().failure(error);
                }
        }

        private boolean isRetryable(Throwable error) {
                return !(error instanceof ValidationException ||
                                error instanceof AuthorizationException ||
                                error instanceof TenantQuotaExceededException);
        }

        private Duration calculateBackoff(int attempt) {
                // Exponential backoff with configured initial and max values
                long initialMs = initialBackoff.toMillis();
                long maxMs = maxBackoff.toMillis();

                long backoffMs = initialMs * (long) Math.pow(2, attempt);
                long cappedMs = Math.min(backoffMs, maxMs);

                return Duration.ofMillis(cappedMs);
        }

        @Override
        public InferenceResponse inferSync(
                        InferenceRequest request,
                        TenantContext tenantContext) {
                LOG.debugf("Starting synchronous inference request: %s", request.getRequestId());
                return infer(request, tenantContext)
                                .await().atMost(syncTimeout);
        }

        @Override
        public Uni<StreamingResponse> inferStream(
                        InferenceRequest request,
                        TenantContext tenantContext) {
                return Uni.createFrom().failure(new UnsupportedOperationException("Streaming not implemented yet"));
        }

        @Override
        public EngineMetadata metadata() {
                return EngineMetadata.builder()
                                .version("2.0.0")
                                .supportedPhases(InferencePhase.ordered())
                                .build();
        }

        @Override
        public HealthStatus health() {
                return HealthStatus.healthy();
        }
}