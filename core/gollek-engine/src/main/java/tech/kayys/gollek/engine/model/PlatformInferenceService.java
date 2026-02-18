package tech.kayys.gollek.engine.model;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.faulttolerance.CircuitBreaker;
import org.eclipse.microprofile.faulttolerance.Retry;
import org.eclipse.microprofile.faulttolerance.Timeout;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.inference.*;
import tech.kayys.gollek.core.inference.InferenceEngine;
import tech.kayys.gollek.engine.inference.InferenceMetrics;
import tech.kayys.gollek.engine.security.EngineQuotaEnforcer;
import tech.kayys.gollek.observability.AuditService;
import jakarta.ws.rs.NotFoundException;

import java.time.temporal.ChronoUnit;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Platform-level inference service with full enterprise features:
 * - Quota enforcement
 * - Circuit breaker
 * - Retry logic
 * - Audit logging
 * - Metrics collection
 */
@ApplicationScoped
public class PlatformInferenceService {

    private static final Logger LOG = Logger.getLogger(PlatformInferenceService.class);

    @Inject
    InferenceEngine engine;

    @Inject
    InferenceMetrics metrics;

    @Inject
    AuditService auditService;

    @Inject
    EngineQuotaEnforcer quotaEnforcer;

    private final Map<String, BatchStatus> batchStatuses = new ConcurrentHashMap<>();

    private UUID parseRequestId(String requestId) {
        if (requestId == null || requestId.isBlank()) {
            return UUID.randomUUID();
        }
        try {
            return UUID.fromString(requestId);
        } catch (IllegalArgumentException e) {
            return UUID.nameUUIDFromBytes(requestId.getBytes());
        }
    }

    /**
     * Execute synchronous inference with full enterprise controls
     */
    @Timeout(value = 30, unit = ChronoUnit.SECONDS)
    @Retry(maxRetries = 2, delay = 1000, jitter = 500)
    @CircuitBreaker(requestVolumeThreshold = 10, failureRatio = 0.5, delay = 5000)
    public Uni<InferenceResponse> infer(
            InferenceRequest request,
            RequestContext requestContext) {
        RequestContext effectiveContext = ensureRequestContext(requestContext, request.getRequestId());

        // 1. Pre-execution quota check
        return quotaEnforcer
                .checkAndIncrementQuota(parseRequestId(effectiveContext.getRequestId()), "requests", 1)
                .onItem().transformToUni(quotaOk -> {
                    if (!quotaOk) {
                        return Uni.createFrom().failure(
                                new RuntimeException("Tenant quota exceeded"));
                    }

                    // 2. Audit start
                    auditService.logInferenceStart(request, effectiveContext);

                    // 3. Execute inference
                    long startTime = System.nanoTime();

                    // Enrich request with context info for downward layers (Providers)
                    InferenceRequest enrichedRequest = request.toBuilder()
                            .userId(effectiveContext.userId())
                            .sessionId(effectiveContext.sessionId())
                            .traceId(effectiveContext.traceId())
                            .metadata("tenantId", effectiveContext.tenantId())
                            .build();

                    return engine.infer(enrichedRequest)
                            .onItem().invoke(response -> {
                                // 4. Record metrics
                                long durationMs = (System.nanoTime() - startTime) / 1_000_000;
                                metrics.recordSuccess(
                                        effectiveContext.getRequestId(),
                                        request.getModel(),
                                        "default",
                                        durationMs);

                                // 5. Audit completion
                                auditService.logInferenceComplete(
                                        request,
                                        response,
                                        effectiveContext);
                            })
                            .onFailure().invoke(error -> {
                                // 6. Record failure metrics
                                metrics.recordFailure(
                                        effectiveContext.getRequestId(),
                                        request.getModel(),
                                        error.getClass().getSimpleName());

                                // 7. Audit failure
                                auditService.logInferenceFailure(
                                        request,
                                        error,
                                        effectiveContext);
                            });
                });
    }

    /**
     * Execute streaming inference
     */
    public Multi<StreamingInferenceChunk> stream(
            InferenceRequest request,
            RequestContext requestContext) {
        RequestContext effectiveContext = ensureRequestContext(requestContext, request.getRequestId());
        // Quota check
        return quotaEnforcer
                .checkAndIncrementQuota(parseRequestId(effectiveContext.getRequestId()), "requests", 1)
                .onItem().transformToMulti(quotaOk -> {
                    if (!quotaOk) {
                        return Multi.createFrom().failure(
                                new RuntimeException("Tenant quota exceeded"));
                    }

                    auditService.logStreamStart(request, effectiveContext);

                    InferenceRequest enrichedRequest = request.toBuilder()
                            .userId(effectiveContext.userId())
                            .sessionId(effectiveContext.sessionId())
                            .traceId(effectiveContext.traceId())
                            .metadata("tenantId", effectiveContext.tenantId())
                            .build();

                    return engine.stream(enrichedRequest)
                            .map(chunk -> (StreamingInferenceChunk) chunk)
                            .onItem().invoke(chunk -> {
                                metrics.recordRequestStarted(
                                        effectiveContext.getRequestId(),
                                        request.getModel());
                            })
                            .onCompletion().invoke(() -> {
                                auditService.logStreamComplete(request, effectiveContext);
                            })
                            .onFailure().invoke(error -> {
                                auditService.logStreamFailure(
                                        request,
                                        error,
                                        effectiveContext);
                            });
                });
    }

    /**
     * Submit asynchronous inference job
     */
    public Uni<String> submitAsyncJob(
            InferenceRequest request,
            RequestContext requestContext) {
        RequestContext effectiveContext = ensureRequestContext(requestContext, request.getRequestId());
        // Quota check
        return quotaEnforcer
                .checkAndIncrementQuota(parseRequestId(effectiveContext.getRequestId()), "requests", 1)
                .onItem().transformToUni(quotaOk -> {
                    if (!quotaOk) {
                        return Uni.createFrom().failure(
                                new RuntimeException("Tenant quota exceeded"));
                    }

                    // Submit to engine's async handler
                    InferenceRequest enrichedRequest = request.toBuilder()
                            .userId(effectiveContext.userId())
                            .sessionId(effectiveContext.sessionId())
                            .traceId(effectiveContext.traceId())
                            .metadata("tenantId", effectiveContext.tenantId())
                            .build();
                    return engine.submitAsyncJob(enrichedRequest);
                });
    }

    /**
     * Execute batch inference
     */
    public Uni<String> batchInfer(
            BatchInferenceRequest batchRequest,
            RequestContext requestContext) {
        RequestContext effectiveContext = ensureRequestContext(requestContext, UUID.randomUUID().toString());
        String batchId = UUID.randomUUID().toString();

        BatchStatus status = new BatchStatus(
                batchId,
                batchRequest.getRequests().size(),
                "PENDING");
        batchStatuses.put(batchId, status);

        // Process batch asynchronously
        Multi.createFrom().iterable(batchRequest.getRequests())
                .onItem().transformToUniAndConcatenate(req -> infer(req, effectiveContext)
                        .onItem().invoke(resp -> status.incrementCompleted())
                        .onFailure().invoke(err -> status.incrementFailed()))
                .subscribe().with(
                        item -> {
                        },
                        error -> {
                            LOG.errorf(error, "Batch failed: %s", batchId);
                            status.setStatus("FAILED");
                        },
                        () -> {
                            LOG.infof("Batch completed: %s", batchId);
                            status.setStatus("COMPLETED");
                        });

        return Uni.createFrom().item(batchId);
    }

    /**
     * Get batch status
     */
    public Uni<BatchStatus> getBatchStatus(
            String batchId,
            RequestContext requestContext) {
        ensureRequestContext(requestContext, batchId);
        BatchStatus status = batchStatuses.get(batchId);
        if (status == null) {
            return Uni.createFrom().failure(
                    new NotFoundException("Batch not found: " + batchId));
        }
        return Uni.createFrom().item(status);
    }

    /**
     * Cancel inference request
     */
    public Uni<Boolean> cancel(
            String requestId,
            RequestContext requestContext) {
        RequestContext effectiveContext = ensureRequestContext(requestContext, requestId);
        // Implementation depends on execution model
        // For now, just log
        LOG.infof("Cancellation requested for: %s", requestId);
        auditService.logCancellation(requestId, effectiveContext);
        return Uni.createFrom().item(true);
    }

    private RequestContext ensureRequestContext(RequestContext requestContext, String requestId) {
        if (requestContext != null) {
            return requestContext;
        }
        return RequestContext.of(requestId);
    }

    /**
     * Batch status tracking
     */
    public static class BatchStatus {
        private final String batchId;
        private final int total;
        private int completed = 0;
        private int failed = 0;
        private String status;

        public BatchStatus(String batchId, int total, String status) {
            this.batchId = batchId;
            this.total = total;
            this.status = status;
        }

        public synchronized void incrementCompleted() {
            completed++;
        }

        public synchronized void incrementFailed() {
            failed++;
        }

        public synchronized void setStatus(String status) {
            this.status = status;
        }

        // Getters
        public String getBatchId() {
            return batchId;
        }

        public int getTotal() {
            return total;
        }

        public int getCompleted() {
            return completed;
        }

        public int getFailed() {
            return failed;
        }

        public String getStatus() {
            return status;
        }
    }
}
