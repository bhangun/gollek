package tech.kayys.golek.engine.inference;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.awssdk.core.exception.RetryableException;
import tech.kayys.golek.spi.inference.AsyncJobStatus;
import tech.kayys.golek.spi.inference.BatchInferenceRequest;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.exception.InferenceException;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.inference.StreamingInferenceChunk;
import tech.kayys.golek.spi.inference.StreamingSession;
import tech.kayys.golek.spi.inference.ValidationContext;
import tech.kayys.golek.engine.model.Model;
import tech.kayys.golek.engine.model.ModelRegistryService;

import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.spi.error.ErrorCode;
import tech.kayys.golek.engine.service.AsyncJobManager;
import tech.kayys.wayang.tenant.QuotaEnforcer;
import tech.kayys.wayang.tenant.Tenant;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.eclipse.microprofile.faulttolerance.Bulkhead;
import org.eclipse.microprofile.faulttolerance.CircuitBreaker;
import org.eclipse.microprofile.faulttolerance.Retry;
import org.eclipse.microprofile.faulttolerance.Timeout;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Core inference service implementation.
 */
@ApplicationScoped
public class InferenceService {

    private static final Logger log = LoggerFactory.getLogger(InferenceService.class);

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

    @ConfigProperty(name = "wayang.multitenancy.enabled", defaultValue = "false")
    boolean multitenancyEnabled;

    private final Map<String, StreamingSession> streamingSessions = new ConcurrentHashMap<>();

    /**
     * Execute synchronous inference with full validation and audit logging.
     */
    @Timeout(value = 30, unit = ChronoUnit.SECONDS)
    @Retry(maxRetries = 2, delay = 100, delayUnit = ChronoUnit.MILLIS, retryOn = { RetryableException.class })
    @CircuitBreaker(requestVolumeThreshold = 20, failureRatio = 0.5, delay = 5000, successThreshold = 2)
    @Bulkhead(value = 100, waitingTaskQueue = 50)
    public Uni<InferenceResponse> inferAsync(InferenceRequest request) {
        return inferAsync(request, null);
    }

    public Uni<InferenceResponse> inferAsync(InferenceRequest request, TenantContext tenantContext) {
        String requestId = request.getRequestId();
        String tenantId = resolveTenantId(tenantContext);

        log.info("Processing inference request: requestId={}, tenantId={}, model={}",
                requestId, tenantId, request.getModel());

        if (!multitenancyEnabled) {
            return orchestrator
                    .executeAsync(request.getModel(), request, TenantContext.of(tenantId))
                    .onItem().invoke(response -> {
                        metrics.recordSuccess(tenantId, request.getModel(), "unified",
                                response.getDurationMs());
                        log.info("Inference completed: requestId={}, durationMs={}, model={}",
                                requestId, response.getDurationMs(), response.getModel());
                    })
                    .onFailure().invoke(failure -> {
                        metrics.recordFailure(tenantId, request.getModel(),
                                failure.getClass().getSimpleName());
                        log.error("Inference failed: requestId={}", requestId, failure);
                    });
        }

        return validateTenant(tenantId)
                .chain(tenant -> validateModel(tenantId, request.getModel())
                        .map(model -> new ValidationContext(tenant.tenantId, model.modelId)))
                .chain((ValidationContext ctx) -> enforceQuota(tenantId, request)
                        .chain(() -> createAuditRecord(request, null, null,
                                InferenceRequestEntity.RequestStatus.PROCESSING))
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
                                })));
    }

    /**
     * Submit async inference job and return immediately.
     */
    public Uni<String> submitAsyncJob(InferenceRequest request) {
        return submitAsyncJob(request, null);
    }

    public Uni<String> submitAsyncJob(InferenceRequest request, TenantContext tenantContext) {
        String jobId = UUID.randomUUID().toString();
        String tId = resolveTenantId(tenantContext);

        log.info("Submitting async job: jobId={}, requestId={}, model={}",
                jobId, request.getRequestId(), request.getModel());

        if (!multitenancyEnabled) {
            asyncJobManager.enqueue(jobId, request, tId);
            return Uni.createFrom().item(jobId);
        }

        return validateTenant(tId)
                .chain(tenant -> enforceQuota(tId, request).replaceWith(tenant))
                .map(tenant -> {
                    asyncJobManager.enqueue(jobId, request, tId);
                    return jobId;
                });
    }

    /**
     * Get status of async inference job.
     */
    public Uni<AsyncJobStatus> getJobStatus(String jobId, String tenantId) {
        AsyncJobStatus status = asyncJobManager.getStatus(jobId);
        if (status == null) {
            return Uni.createFrom().failure(new InferenceException(ErrorCode.INTERNAL_ERROR, "Job not found: " + jobId)
                    .addContext("jobId", jobId));
        }

        if (!multitenancyEnabled) {
            return Uni.createFrom().item(status);
        }

        if (!status.tenantId().equals(tenantId)) {
            return Uni.createFrom().failure(
                    new InferenceException(ErrorCode.AUTH_PERMISSION_DENIED, "Access denied to job: " + jobId));
        }

        return Uni.createFrom().item(status);
    }

    /**
     * Stream inference results for generative models.
     */
    public Multi<StreamingInferenceChunk> inferStream(InferenceRequest request) {
        return inferStream(request, null);
    }

    public Multi<StreamingInferenceChunk> inferStream(InferenceRequest request, TenantContext tenantContext) {
        String requestId = request.getRequestId();
        String tId = resolveTenantId(tenantContext);

        return Multi.createFrom().emitter(emitter -> {
            if (!multitenancyEnabled) {
                try {
                    StreamingSession session = new StreamingSession(requestId, request.getModel(), "community",
                            Multi.createFrom().empty());
                    streamingSessions.put(requestId, session);

                    java.util.concurrent.atomic.AtomicInteger sequenceNumber = new java.util.concurrent.atomic.AtomicInteger(
                            0);
                    long startTime = System.currentTimeMillis();

                    orchestrator
                            .streamExecute(request.getModel(), request, TenantContext.of(tId))
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
                return;
            }

            validateTenant(tId)
                    .chain(tenant -> validateModel(tId, request.getModel())
                            .map(model -> new ValidationContext(tenant.tenantId, model.modelId)))
                    .chain(ctx -> enforceQuota(tId, request).replaceWith(ctx))
                    .subscribe().with(
                            (ValidationContext ctx) -> {
                                try {
                                    StreamingSession session = new StreamingSession(requestId, request.getModel(), tId,
                                            Multi.createFrom().empty());
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
        String tId = resolveTenantId(tenantId);
        log.info("Processing batch inference: size={}, tenantId={}", batchRequest.getRequests().size(), tId);

        return Multi.createFrom().items(batchRequest.getRequests().stream())
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
                .onItem().ifNull()
                .failWith(
                        () -> new InferenceException(ErrorCode.AUTH_TENANT_NOT_FOUND, "Tenant not found: " + tenantId))
                .invoke(tenant -> {
                    if (tenant.status != Tenant.TenantStatus.ACTIVE) {
                        throw new InferenceException("Tenant is not active: " + tenantId);
                    }
                });
    }

    private Uni<Void> enforceQuota(String tenantId, InferenceRequest request) {
        // Resolve UUID from tenantId if possible, or just use a fixed one for now if
        // tenantId is "community"
        UUID tenantUuid = UUID.nameUUIDFromBytes(tenantId.getBytes());
        return quotaEnforcer.checkAndIncrementQuota(tenantUuid, "inference", 1L)
                .onItem().invoke(allowed -> {
                    if (!allowed) {
                        throw new InferenceException("Quota exceeded for tenant: " + tenantId);
                    }
                })
                .replaceWithVoid();
    }

    private Uni<Model> validateModel(String tenantId, String modelId) {
        String modelName = modelId.split(":")[0];
        return Model.findByTenantAndModelId(tenantId, modelName)
                .onItem().ifNull()
                .failWith(() -> new InferenceException(ErrorCode.MODEL_NOT_FOUND, "Model not found: " + modelId));
    }

    private String resolveTenantId(TenantContext tenantContext) {
        if (tenantContext == null || tenantContext.getTenantId() == null) {
            return "community";
        }
        String tenantId = tenantContext.getTenantId().value();
        return (tenantId == null || tenantId.trim().isEmpty()) ? "community" : tenantId;
    }

    private String resolveTenantId(String tenantId) {
        return (tenantId == null || tenantId.trim().isEmpty()) ? "community" : tenantId;
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

}
