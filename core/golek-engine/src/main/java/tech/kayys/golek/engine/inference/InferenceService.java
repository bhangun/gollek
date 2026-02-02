package tech.kayys.golek.engine.inference;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import lombok.extern.slf4j.Slf4j;
import software.amazon.awssdk.core.exception.RetryableException;
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.exception.InferenceException;
import tech.kayys.golek.engine.inference.InferenceResource.StreamingInferenceChunk;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.engine.model.Model;
import tech.kayys.golek.engine.model.ModelRegistryService;
import tech.kayys.golek.engine.tenant.AuthenticationException;
import tech.kayys.golek.engine.tenant.QuotaEnforcer;
import tech.kayys.golek.engine.tenant.Tenant;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.api.error.ErrorCode;
import tech.kayys.golek.api.exception.ModelException;

import org.eclipse.microprofile.faulttolerance.Bulkhead;
import org.eclipse.microprofile.faulttolerance.CircuitBreaker;
import org.eclipse.microprofile.faulttolerance.Retry;
import org.eclipse.microprofile.faulttolerance.Timeout;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Core inference service implementation.
 */
@ApplicationScoped
@Slf4j
public class InferenceService {

    @Inject
    InferenceOrchestrator orchestrator;

    @Inject
    QuotaEnforcer quotaEnforcer;

    @Inject
    InferenceMetrics metrics;

    @Inject
    AsyncJobManager asyncJobManager;

    @Inject
    ModelRegistryService modelRegistry;

    private final Map<String, StreamingSession> streamingSessions = new ConcurrentHashMap<>();

    /**
     * Execute synchronous inference with full validation and audit logging.
     */
    @Timeout(value = 30, unit = ChronoUnit.SECONDS)
    @Retry(maxRetries = 2, delay = 100, delayUnit = ChronoUnit.MILLIS, retryOn = { RetryableException.class })
    @CircuitBreaker(requestVolumeThreshold = 20, failureRatio = 0.5, delay = 5000, successThreshold = 2)
    @Bulkhead(value = 100, waitingTaskQueue = 50)
    public Uni<InferenceResponse> inferAsync(InferenceRequest request) {
        String requestId = request.getRequestId();
        String tenantId = request.getTenantId();

        log.info("Processing inference request: requestId={}, tenantId={}, model={}",
                requestId, tenantId, request.getModel());

        return validateTenant(tenantId)
                .chain(tenant -> validateModel(tenantId, request.getModel())
                        .map(model -> new ValidationContext(tenant, model)))
                .chain(ctx -> {
                    enforceQuota(ctx.tenant, request);

                    return createAuditRecord(request, ctx.tenant, ctx.model,
                            InferenceRequestEntity.RequestStatus.PROCESSING)
                            .chain(auditRecord -> orchestrator
                                    .executeAsync(request.getModel(), request, TenantContext.of(tenantId))
                                    .onItem().call(response -> updateAuditRecord(auditRecord, response, null))
                                    .onItem().invoke(response -> {
                                        metrics.recordSuccess(tenantId, request.getModel(), "unified",
                                                response.getDurationMs());
                                        log.info("Inference completed: requestId={}, durationMs={}, model={}",
                                                requestId, response.getDurationMs(), response.getModel());
                                    })
                                    .onFailure().call(failure -> updateAuditRecord(auditRecord, null, failure))
                                    .onFailure().invoke(failure -> {
                                        metrics.recordFailure(tenantId, request.getModel(),
                                                failure.getClass().getSimpleName());
                                        log.error("Inference failed: requestId={}", requestId, failure);
                                    }));
                });
    }

    /**
     * Submit async inference job and return immediately.
     */
    public Uni<String> submitAsyncJob(InferenceRequest request) {
        String jobId = UUID.randomUUID().toString();
        String tId = request.getTenantId() != null ? request.getTenantId() : "default";

        log.info("Submitting async job: jobId={}, requestId={}, model={}",
                jobId, request.getRequestId(), request.getModel());

        return validateTenant(tId)
                .invoke(tenant -> enforceQuota(tenant, request))
                .map(tenant -> {
                    asyncJobManager.enqueue(jobId, request);
                    return jobId;
                });
    }

    /**
     * Get status of async inference job.
     */
    public Uni<AsyncJobStatus> getJobStatus(String jobId, String tenantId) {
        AsyncJobStatus status = asyncJobManager.getStatus(jobId);
        if (status == null) {
            return Uni.createFrom().failure(new InferenceException("Job not found: " + jobId, jobId));
        }

        if (!status.tenantId().equals(tenantId)) {
            return Uni.createFrom().failure(new InferenceException("Access denied to job: " + jobId));
        }

        return Uni.createFrom().item(status);
    }

    /**
     * Stream inference results for generative models.
     */
    public Multi<StreamingInferenceChunk> inferStream(InferenceRequest request) {
        String requestId = request.getRequestId();
        String tId = request.getTenantId() != null ? request.getTenantId() : "default";

        return Multi.createFrom().emitter(emitter -> {
            validateTenant(tId)
                    .chain(tenant -> validateModel(tId, request.getModel())
                            .map(model -> new ValidationContext(tenant, model)))
                    .subscribe().with(
                            ctx -> {
                                try {
                                    enforceQuota(ctx.tenant, request);
                                    StreamingSession session = new StreamingSession(requestId, request);
                                    streamingSessions.put(requestId, session);

                                    java.util.concurrent.atomic.AtomicInteger sequenceNumber = new java.util.concurrent.atomic.AtomicInteger(
                                            0);
                                    long startTime = System.currentTimeMillis();

                                    orchestrator
                                            .streamExecute(request.getModel(), request,
                                                    tech.kayys.wayang.tenant.TenantContext.of(tId))
                                            .subscribe().with(
                                                    chunk -> {
                                                        sequenceNumber.getAndIncrement();
                                                        emitter.emit(new StreamingInferenceChunk(
                                                                requestId,
                                                                sequenceNumber.get(),
                                                                chunk.getDelta(),
                                                                chunk.isFinal(),
                                                                System.currentTimeMillis() - startTime));
                                                    },
                                                    failure -> {
                                                        log.error("Streaming inference failed: requestId={}", requestId,
                                                                failure);
                                                        emitter.fail(mapToInferenceException(failure));
                                                        streamingSessions.remove(requestId);
                                                    },
                                                    () -> {
                                                        emitter.complete();
                                                        streamingSessions.remove(requestId);
                                                    });
                                } catch (Exception e) {
                                    emitter.fail(mapToInferenceException(e));
                                }
                            },
                            failure -> emitter.fail(mapToInferenceException(failure)));
        });
    }

    /**
     * Batch inference - process multiple requests.
     */
    public Uni<List<InferenceResponse>> batchInfer(BatchInferenceRequest batchRequest, String tenantId) {
        log.info("Processing batch inference: size={}, tenantId={}", batchRequest.requests().size(), tenantId);

        return Multi.createFrom().items(batchRequest.requests().stream())
                .onItem().transformToUniAndConcatenate(request -> inferAsync(request)
                        .onFailure().recoverWithItem(failure -> InferenceResponse.builder()
                                .requestId(request.getRequestId())
                                .model(request.getModel())
                                .content("Error: " + failure.getMessage())
                                .build()))
                .collect().asList();
    }

    // ===== Helper Methods =====

    private Uni<Tenant> validateTenant(String tenantId) {
        return Tenant.findByTenantId(tenantId)
                .onItem().ifNull().failWith(() -> new InferenceException("Tenant not found: " + tenantId))
                .invoke(tenant -> {
                    if (tenant.status != Tenant.TenantStatus.ACTIVE) {
                        throw new InferenceException("Tenant is not active: " + tenantId);
                    }
                });
    }

    private void enforceQuota(Tenant tenant, InferenceRequest request) {
        if (!quotaEnforcer.checkAndIncrementQuota(tenant.tenantId, "requests", 1)) {
            throw new InferenceException("Quota exceeded for tenant: " + tenant.tenantId);
        }
    }

    private Uni<Model> validateModel(String tenantId, String modelId) {
        String modelName = modelId.split(":")[0];
        return Model.findByTenantAndModelId(tenantId, modelName)
                .onItem().ifNull().failWith(() -> new InferenceException("Model not found: " + modelId, modelId));
    }

    @Transactional
    public Uni<InferenceRequestEntity> createAuditRecord(InferenceRequest request, Tenant tenant, Model model,
            InferenceRequestEntity.RequestStatus status) {
        InferenceRequestEntity entity = InferenceRequestEntity.builder()
                .requestId(request.getRequestId())
                .tenant(tenant)
                .model(model)
                .status(status)
                .inputSizeBytes(calculateInputSize(request))
                .metadata(Map.of("parameters", request.getParameters()))
                .createdAt(LocalDateTime.now())
                .build();

        return entity.persist().replaceWith(entity);
    }

    @Transactional
    public Uni<Void> updateAuditRecord(InferenceRequestEntity entity, InferenceResponse response, Throwable failure) {
        if (response != null) {
            entity.status = InferenceRequestEntity.RequestStatus.COMPLETED;
            entity.runnerName = "unified";
            entity.latencyMs = response.getDurationMs();
            entity.outputSizeBytes = (long) response.getContent().length();
        } else if (failure != null) {
            entity.status = InferenceRequestEntity.RequestStatus.FAILED;
            entity.errorMessage = failure.getMessage();
        }

        entity.completedAt = LocalDateTime.now();
        return entity.persist().replaceWithVoid();
    }

    private long calculateInputSize(InferenceRequest request) {
        return request.getMessages().stream()
                .mapToLong(msg -> msg.getContent() != null ? msg.getContent().length() : 0)
                .sum();
    }

    private InferenceException mapToInferenceException(Throwable t) {
        if (t instanceof InferenceException ie)
            return ie;
        return new InferenceException("Internal error: " + t.getMessage(), t);
    }

    // ===== Inner Classes =====

    private record ValidationContext(Tenant tenant, Model model) {
    }

    private static class StreamingSession {
        private final String requestId;
        private final InferenceRequest request;
        private final Instant startTime;
        private volatile boolean active = true;

        public StreamingSession(String requestId, InferenceRequest request) {
            this.requestId = requestId;
            this.request = request;
            this.startTime = Instant.now();
        }

        public boolean isActive() {
            return active;
        }

        public void cancel() {
            this.active = false;
        }
    }

    public record BatchInferenceRequest(List<InferenceRequest> requests, Integer maxConcurrent) {
    }

    public record AsyncJobStatus(String jobId, String requestId, String tenantId, String status,
            InferenceResponse result, String error, Instant submittedAt, Instant completedAt) {
        public boolean isComplete() {
            return "COMPLETED".equals(status) || "FAILED".equals(status);
        }
    }
}
