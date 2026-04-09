package tech.kayys.gollek.engine.inference;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.awssdk.core.exception.RetryableException;
import tech.kayys.gollek.spi.inference.AsyncJobStatus;
import tech.kayys.gollek.spi.inference.InferenceEngine;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingSession;
import tech.kayys.gollek.spi.inference.ValidationContext;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

import tech.kayys.gollek.spi.batch.BatchInferenceRequest;
import tech.kayys.gollek.spi.batch.BatchResponse;
import tech.kayys.gollek.spi.batch.BatchScheduler;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.engine.service.AsyncJobManager;
import tech.kayys.gollek.engine.context.DevicePreferenceResolver;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.registry.model.Model;
import tech.kayys.gollek.registry.service.ModelRegistryService;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.engine.security.EngineQuotaEnforcer;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.eclipse.microprofile.faulttolerance.Bulkhead;
import org.eclipse.microprofile.faulttolerance.CircuitBreaker;
import org.eclipse.microprofile.faulttolerance.Retry;
import org.eclipse.microprofile.faulttolerance.Timeout;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Core inference service implementation.
 */
@ApplicationScoped
public class InferenceService {

    private static final Logger log = LoggerFactory.getLogger(InferenceService.class);

    @Inject
    InferenceEngine inferenceEngine;

    @Inject
    InferenceOrchestrator orchestrator;

    @Inject
    BatchScheduler batchScheduler;

    @Inject
    EngineQuotaEnforcer quotaEnforcer;

    @Inject
    InferenceMetrics metrics;

    @Inject
    AsyncJobManager asyncJobManager;

    @Inject
    DevicePreferenceResolver devicePreferenceResolver;

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

    public Uni<InferenceResponse> inferAsync(InferenceRequest request, RequestContext requestContext) {
        RequestContext effectiveContext = resolveContext(requestContext, request);
        String tenantId = effectiveContext.getRequestId();

        log.info("Processing inference request: tenantId={}, model={}",
                tenantId, request.getModel());

        if (!multitenancyEnabled) {
            return orchestrator
                    .executeAsync(request.getModel(), request)
                    .onItem().invoke(response -> {
                        metrics.recordSuccess(tenantId, request.getModel(), "unified",
                                response.getDurationMs());
                        log.info("Inference completed: tenantId={}, durationMs={}, model={}",
                                tenantId, response.getDurationMs(), response.getModel());
                    })
                    .onFailure().invoke(failure -> {
                        metrics.recordFailure(tenantId, request.getModel(),
                                failure.getClass().getSimpleName());
                        log.error("Inference failed: tenantId={}", tenantId, failure);
                    });
        }

        return validateTenant(tenantId, request.getRequestId())
                .chain(tenant -> validateModel(tenant.tenantId, request.getModel())
                        .map(model -> new ValidationContext(tenant.tenantId, model.modelId)))
                .chain((ValidationContext ctx) -> enforceQuota(tenantId, request)
                        .chain(() -> createAuditRecord(request, null,
                                InferenceRequestEntity.RequestStatus.PROCESSING))
                        .chain(auditRecord -> orchestrator
                                .executeAsync(request.getModel(), request)
                                .onItem().call(response -> updateAuditRecord(auditRecord, response, null))
                                .onItem().invoke(response -> {
                                    metrics.recordSuccess(tenantId, request.getModel(), "unified",
                                            response.getDurationMs());
                                    log.info("Inference completed: tenantId={}, durationMs={}, model={}",
                                            tenantId, response.getDurationMs(), response.getModel());
                                })
                                .onFailure().call(failure -> updateAuditRecord(auditRecord, null, failure))
                                .onFailure().invoke(failure -> {
                                    metrics.recordFailure(tenantId, request.getModel(),
                                            failure.getClass().getSimpleName());
                                    log.error("Inference failed: tenantId={}", tenantId, failure);
                                })));
    }

    /**
     * Submit async inference job and return immediately.
     */
    public Uni<String> submitAsyncJob(InferenceRequest request) {
        return submitAsyncJob(request, null);
    }

    public Uni<tech.kayys.gollek.spi.embedding.EmbeddingResponse> executeEmbedding(
            tech.kayys.gollek.spi.embedding.EmbeddingRequest request) {
        return inferenceEngine.executeEmbedding(request.model(), request);
    }

    public Uni<String> submitAsyncJob(InferenceRequest request, RequestContext requestContext) {
        String jobId = UUID.randomUUID().toString();
        RequestContext effectiveContext = resolveContext(requestContext, request);
        String tId = effectiveContext.getRequestId();

        log.info("Submitting async job: jobId={}, requestId={}, model={}",
                jobId, request.getRequestId(), request.getModel());

        if (!multitenancyEnabled) {
            asyncJobManager.enqueue(jobId, request, tId);
            return Uni.createFrom().item(jobId);
        }

        return validateTenant(tId, request.getRequestId())
                .chain(tenant -> enforceQuota(tId, request).replaceWith(tenant))
                .map(tenant -> {
                    asyncJobManager.enqueue(jobId, request, tId);
                    return jobId;
                });
    }

    /**
     * Get status of async inference job.
     */
    public Uni<AsyncJobStatus> getJobStatus(String jobId, String requestId) {
        AsyncJobStatus status = asyncJobManager.getStatus(jobId);
        if (status == null) {
            return Uni.createFrom().failure(new InferenceException(ErrorCode.INTERNAL_ERROR, "Job not found: " + jobId)
                    .addContext("jobId", jobId));
        }

        if (!multitenancyEnabled) {
            return Uni.createFrom().item(status);
        }

        if (!status.requestId().equals(requestId)) {
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

    public Multi<StreamingInferenceChunk> inferStream(InferenceRequest request, RequestContext requestContext) {
        RequestContext effectiveContext = resolveContext(requestContext, request);
        String tenantId = effectiveContext.getRequestId();

        return Multi.createFrom().emitter(emitter -> {
            if (multitenancyEnabled) {
                validateTenant(tenantId, request.getRequestId())
                        .chain(tenant -> enforceQuota(tenantId, request).replaceWith(tenant))
                        .subscribe().with(
                                tenant -> startStream(request, tenantId, effectiveContext, emitter),
                                failure -> emitter.fail(mapToInferenceException(failure)));
            } else {
                try {
                    startStream(request, tenantId, effectiveContext, emitter);
                } catch (Exception e) {
                    emitter.fail(mapToInferenceException(e));
                }
            }
        });
    }

    /**
     * Batch inference — delegates to the configured {@link BatchScheduler}.
     * <p>
     * The scheduler applies the active batching strategy (STATIC / DYNAMIC /
     * CONTINUOUS)
     * and returns a {@link BatchResponse} with per-request results and aggregate
     * metrics.
     */
    public Uni<BatchResponse> batchInfer(BatchInferenceRequest batchRequest) {
        log.info("Processing batch inference via scheduler: size={}, requestId={}",
                batchRequest.getRequests().size(), batchRequest.modelId());

        return Uni.createFrom().completionStage(
                batchScheduler.submitBatch(batchRequest));
    }

    // ===== Helper Methods =====

    private Uni<Model> validateModel(String requestId, String modelId) {
        String modelName = modelId.split(":")[0];
        return Model.findByTenantAndModelId(requestId, modelName)
                .onItem().ifNull()
                .failWith(() -> new InferenceException(ErrorCode.MODEL_NOT_FOUND, "Model not found: " + modelId));
    }

    @Transactional
    public Uni<InferenceRequestEntity> createAuditRecord(InferenceRequest request, Model model,
            InferenceRequestEntity.RequestStatus status) {
        InferenceRequestEntity entity = InferenceRequestEntity.builder()
                .requestId(request.getRequestId())

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

    private void startStream(
            InferenceRequest request,
            String requestId,
            RequestContext effectiveContext,
            io.smallrye.mutiny.subscription.MultiEmitter<? super StreamingInferenceChunk> emitter) {
        StreamingSession session = new StreamingSession(
                requestId,
                request.getModel(),
                effectiveContext.getRequestId(),
                Multi.createFrom().empty());
        streamingSessions.put(requestId, session);

        java.util.concurrent.atomic.AtomicInteger sequenceNumber = new java.util.concurrent.atomic.AtomicInteger(0);
        java.util.concurrent.atomic.AtomicBoolean firstChunkEmitted = new java.util.concurrent.atomic.AtomicBoolean(
                false);
        long startTime = System.currentTimeMillis();

        metrics.recordRequestStarted(requestId, request.getModel());

        orchestrator
                .streamExecute(request.getModel(), request)
                .subscribe().with(
                        chunk -> {
                            int sequence = sequenceNumber.incrementAndGet();
                            long currentTime = System.currentTimeMillis();

                            if (firstChunkEmitted.compareAndSet(false, true)) {
                                log.debug("Time to first chunk for request %s: %d ms", requestId,
                                        (currentTime - startTime));
                            }

                            if (chunk.finished()) {
                                long totalDuration = currentTime - startTime;
                                StreamingInferenceChunk.ChunkUsage usage = new StreamingInferenceChunk.ChunkUsage(0,
                                        sequence, totalDuration);
                                emitter.emit(StreamingInferenceChunk.finalTextChunk(
                                        requestId,
                                        sequence,
                                        chunk.delta(),
                                        usage));

                                metrics.recordSuccess(requestId, request.getModel(), "unified", totalDuration);
                                log.info("Streaming completed for request %s: duration=%d ms, chunks=%d",
                                        requestId, totalDuration, sequence);
                            } else {
                                emitter.emit(StreamingInferenceChunk.of(
                                        requestId,
                                        sequence,
                                        chunk.delta()));
                            }
                        },
                        failure -> {
                            metrics.recordFailure(requestId, request.getModel(), failure.getClass().getSimpleName());

                            if (isStreamingUnsupported(failure)) {
                                log.info(
                                        "Provider does not support streaming, falling back to single-response mode: requestId={}",
                                        requestId);
                                inferenceEngine.executeAsync(request.getModel(), request)
                                        .subscribe().with(
                                                response -> {
                                                    int sequence = sequenceNumber.incrementAndGet();
                                                    long totalDuration = System.currentTimeMillis() - startTime;
                                                    StreamingInferenceChunk.ChunkUsage usage = new StreamingInferenceChunk.ChunkUsage(
                                                            0, 1, totalDuration);

                                                    emitter.emit(StreamingInferenceChunk.finalTextChunk(
                                                            requestId,
                                                            sequence,
                                                            response.getContent(),
                                                            usage));

                                                    metrics.recordSuccess(requestId, request.getModel(), "unified",
                                                            totalDuration);
                                                    emitter.complete();
                                                    streamingSessions.remove(requestId);
                                                },
                                                fallbackFailure -> {
                                                    log.error("Fallback inference failed: requestId={}", requestId,
                                                            fallbackFailure);
                                                    metrics.recordFailure(requestId, request.getModel(),
                                                            fallbackFailure.getClass().getSimpleName());
                                                    emitter.fail(mapToInferenceException(fallbackFailure));
                                                    streamingSessions.remove(requestId);
                                                });
                                return;
                            }

                            log.error("Streaming inference failed: requestId={}", requestId, failure);
                            emitter.fail(mapToInferenceException(failure));
                            streamingSessions.remove(requestId);
                        },
                        () -> {
                            emitter.complete();
                            streamingSessions.remove(requestId);
                        });
    }

    private RequestContext resolveContext(RequestContext requestContext, InferenceRequest request) {
        RequestContext base = requestContext != null ? requestContext : RequestContext.of(request.getRequestId());
        return devicePreferenceResolver.apply(base, request);
    }

    private Uni<TenantValidation> validateTenant(String tenantId, String requestId) {
        String effectiveTenant = (tenantId == null || tenantId.isBlank())
                ? RequestContext.COMMUNITY_TENANT_ID
                : tenantId.trim();
        String effectiveRequestId = (requestId == null || requestId.isBlank())
                ? UUID.randomUUID().toString()
                : requestId;
        return Uni.createFrom().item(new TenantValidation(effectiveTenant, effectiveRequestId));
    }

    private Uni<Void> enforceQuota(String requestId, InferenceRequest request) {
        return quotaEnforcer
                .checkAndIncrementQuota(parseTenantUuid(requestId), "requests", 1)
                .onItem().transformToUni(allowed -> {
                    if (Boolean.TRUE.equals(allowed)) {
                        return Uni.createFrom().voidItem();
                    }
                    return Uni.createFrom().failure(new InferenceException(
                            ErrorCode.AUTH_PERMISSION_DENIED,
                            "Tenant quota exceeded: " + requestId));
                });
    }

    private UUID parseTenantUuid(String requestId) {
        if (requestId == null || requestId.isBlank()) {
            return UUID.nameUUIDFromBytes(RequestContext.COMMUNITY_TENANT_ID.getBytes());
        }
        try {
            return UUID.fromString(requestId);
        } catch (IllegalArgumentException e) {
            return UUID.nameUUIDFromBytes(requestId.getBytes());
        }
    }

    private record TenantValidation(String tenantId, String requestId) {
    }

    private InferenceException mapToInferenceException(Throwable t) {
        if (t instanceof InferenceException ie)
            return ie;
        return new InferenceException("Internal error: " + t.getMessage(), t);
    }

    private boolean isStreamingUnsupported(Throwable t) {
        // Iterative walk with depth guard to prevent StackOverflow from circular cause
        // chains
        Throwable current = t;
        int maxDepth = 10;
        for (int i = 0; i < maxDepth && current != null; i++) {
            if (current instanceof UnsupportedOperationException) {
                return true;
            }
            String message = current.getMessage();
            if (message != null && message.toLowerCase().contains("does not support streaming")) {
                return true;
            }
            Throwable next = current.getCause();
            if (next == current)
                break; // self-referencing cause chain
            current = next;
        }
        return false;
    }

}
