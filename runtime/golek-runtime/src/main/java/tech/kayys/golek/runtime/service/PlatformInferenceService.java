package tech.kayys.golek.runtime.service;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.faulttolerance.CircuitBreaker;
import org.eclipse.microprofile.faulttolerance.Retry;
import org.eclipse.microprofile.faulttolerance.Timeout;
import org.jboss.logging.Logger;
import tech.kayys.golek.api.inference.*;
import tech.kayys.golek.runtime.InferenceEngine;
import tech.kayys.golek.runtime.audit.AuditService;
import tech.kayys.golek.runtime.metrics.InferenceMetrics;
import tech.kayys.golek.runtime.security.QuotaEnforcer;

import java.time.temporal.ChronoUnit;
import java.util.List;
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
    QuotaEnforcer quotaEnforcer;

    @Inject
    AuditService auditService;

    @Inject
    InferenceMetrics metrics;

    private final Map<String, BatchStatus> batchStatuses = new ConcurrentHashMap<>();

    /**
     * Execute synchronous inference with full enterprise controls
     */
    @Timeout(value = 30, unit = ChronoUnit.SECONDS)
    @Retry(maxRetries = 2, delay = 1000, jitter = 500)
    @CircuitBreaker(requestVolumeThreshold = 10, failureRatio = 0.5, delay = 5000)
    public Uni<InferenceResponse> infer(
            InferenceRequest request,
            TenantContext tenantContext) {
        return Uni.createFrom().deferred(() -> {
            // 1. Pre-execution quota check
            return quotaEnforcer.checkQuota(tenantContext, request)
                    .onItem().transformToUni(quotaOk -> {
                        if (!quotaOk) {
                            return Uni.createFrom().failure(
                                    new QuotaExceededException("Tenant quota exceeded"));
                        }

                        // 2. Audit start
                        auditService.logInferenceStart(request, tenantContext);

                        // 3. Execute inference
                        long startTime = System.nanoTime();

                        return engine.infer(request)
                                .onItem().invoke(response -> {
                                    // 4. Record metrics
                                    long duration = System.nanoTime() - startTime;
                                    metrics.recordSuccess(
                                            request.getModel(),
                                            tenantContext.getTenantId(),
                                            duration,
                                            response.getTokensUsed());

                                    // 5. Consume quota
                                    quotaEnforcer.consumeQuota(
                                            tenantContext,
                                            response.getTokensUsed());

                                    // 6. Audit completion
                                    auditService.logInferenceComplete(
                                            request,
                                            response,
                                            tenantContext);
                                })
                                .onFailure().invoke(error -> {
                                    // 7. Record failure metrics
                                    metrics.recordFailure(
                                            request.getModel(),
                                            tenantContext.getTenantId(),
                                            error.getClass().getSimpleName());

                                    // 8. Audit failure
                                    auditService.logInferenceFailure(
                                            request,
                                            error,
                                            tenantContext);
                                });
                    });
        });
    }

    /**
     * Execute streaming inference
     */
    public Multi<StreamChunk> stream(
            InferenceRequest request,
            TenantContext tenantContext) {
        return Multi.createFrom().deferred(() -> {
            // Quota check
            return quotaEnforcer.checkQuota(tenantContext, request)
                    .onItem().transformToMulti(quotaOk -> {
                        if (!quotaOk) {
                            return Multi.createFrom().failure(
                                    new QuotaExceededException("Tenant quota exceeded"));
                        }

                        auditService.logStreamStart(request, tenantContext);

                        return engine.stream(request)
                                .onItem().invoke(chunk -> {
                                    metrics.recordStreamChunk(
                                            request.getModel(),
                                            tenantContext.getTenantId());
                                })
                                .onCompletion().invoke(() -> {
                                    auditService.logStreamComplete(request, tenantContext);
                                })
                                .onFailure().invoke(error -> {
                                    auditService.logStreamFailure(
                                            request,
                                            error,
                                            tenantContext);
                                });
                    });
        });
    }

    /**
     * Execute batch inference
     */
    public Uni<String> batchInfer(
            BatchInferenceRequest batchRequest,
            TenantContext tenantContext) {
        String batchId = UUID.randomUUID().toString();

        BatchStatus status = new BatchStatus(
                batchId,
                batchRequest.getRequests().size(),
                "PENDING");
        batchStatuses.put(batchId, status);

        // Process batch asynchronously
        Multi.createFrom().iterable(batchRequest.getRequests())
                .onItem().transformToUniAndConcatenate(req -> infer(req, tenantContext)
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
            TenantContext tenantContext) {
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
            TenantContext tenantContext) {
        // Implementation depends on execution model
        // For now, just log
        LOG.infof("Cancellation requested for: %s", requestId);
        auditService.logCancellation(requestId, tenantContext);
        return Uni.createFrom().item(true);
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