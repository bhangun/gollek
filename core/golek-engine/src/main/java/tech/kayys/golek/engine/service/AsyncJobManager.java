package tech.kayys.golek.engine.service;

import io.quarkus.redis.datasource.RedisDataSource;
import io.quarkus.redis.datasource.hash.HashCommands;
import io.quarkus.scheduler.Scheduled;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import lombok.extern.slf4j.Slf4j;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.engine.inference.InferenceOrchestrator;
import tech.kayys.golek.engine.inference.InferenceService.AsyncJobStatus;
import tech.kayys.wayang.tenant.TenantContext;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;

/**
 * Manager for asynchronous inference jobs.
 * 
 * <p>
 * Features:
 * <ul>
 * <li>Job queue with priority support</li>
 * <li>Background worker threads</li>
 * <li>Job status tracking (Redis-backed)</li>
 * <li>Result caching with TTL</li>
 * <li>Automatic cleanup of old jobs</li>
 * </ul>
 * 
 * @author bhangun
 * @since 1.0.0
 */
@ApplicationScoped
@Slf4j
public class AsyncJobManager {

    @Inject
    InferenceOrchestrator orchestrator;

    @Inject
    RedisDataSource redisDataSource;

    private HashCommands<String, String, String> hashCommands;

    // Job queue with priority
    private final PriorityBlockingQueue<AsyncJob> jobQueue = new PriorityBlockingQueue<>(
            1000,
            Comparator.comparingInt(job -> -job.priority) // Higher priority first
    );

    // In-memory job status cache (fallback if Redis unavailable)
    private final ConcurrentHashMap<String, AsyncJobStatus> statusCache = new ConcurrentHashMap<>();

    // Worker thread pool
    private ExecutorService workerPool;

    private static final String REDIS_JOB_PREFIX = "async:job:";
    private static final int DEFAULT_WORKERS = 4;
    private static final int JOB_TTL_HOURS = 24;

    @jakarta.annotation.PostConstruct
    void init() {
        if (redisDataSource != null) {
            this.hashCommands = redisDataSource.hash(String.class);
        }

        // Start worker threads
        int numWorkers = Runtime.getRuntime().availableProcessors();
        this.workerPool = Executors.newFixedThreadPool(
                Math.min(numWorkers, DEFAULT_WORKERS),
                new ThreadFactory() {
                    private int count = 0;

                    public Thread newThread(Runnable r) {
                        return new Thread(r, "async-worker-" + (count++));
                    }
                });

        // Start workers
        for (int i = 0; i < Math.min(numWorkers, DEFAULT_WORKERS); i++) {
            workerPool.submit(this::processJobs);
        }

        log.info("AsyncJobManager initialized with {} workers", Math.min(numWorkers, DEFAULT_WORKERS));
    }

    @jakarta.annotation.PreDestroy
    void shutdown() {
        if (workerPool != null) {
            workerPool.shutdown();
            try {
                if (!workerPool.awaitTermination(30, TimeUnit.SECONDS)) {
                    workerPool.shutdownNow();
                }
            } catch (InterruptedException e) {
                workerPool.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Enqueue async inference job.
     */
    public void enqueue(String jobId, InferenceRequest request) {
        AsyncJob job = new AsyncJob(
                jobId,
                request,
                request.getPriority(),
                Instant.now());

        jobQueue.offer(job);

        // Store initial status
        AsyncJobStatus status = new AsyncJobStatus(
                jobId,
                request.getRequestId(),
                request.getTenantId(),
                "PENDING",
                null,
                null,
                Instant.now(),
                null);

        storeStatus(jobId, status);

        log.info("Job enqueued: jobId={}, priority={}, queueSize={}",
                jobId, request.getPriority(), jobQueue.size());
    }

    /**
     * Get job status.
     */
    public AsyncJobStatus getStatus(String jobId) {
        // Try Redis first
        if (hashCommands != null) {
            try {
                Map<String, String> jobData = hashCommands.hgetall(REDIS_JOB_PREFIX + jobId);
                if (!jobData.isEmpty()) {
                    return parseJobStatus(jobData);
                }
            } catch (Exception e) {
                log.warn("Failed to get job status from Redis", e);
            }
        }

        // Fallback to in-memory cache
        return statusCache.get(jobId);
    }

    /**
     * Cancel pending job.
     */
    public boolean cancelJob(String jobId) {
        AsyncJobStatus status = getStatus(jobId);
        if (status == null || status.isComplete()) {
            return false;
        }

        // Remove from queue if pending
        jobQueue.removeIf(job -> job.jobId.equals(jobId));

        // Update status
        AsyncJobStatus cancelledStatus = new AsyncJobStatus(
                jobId,
                status.requestId(),
                status.tenantId(),
                "CANCELLED",
                null,
                "Job cancelled by user",
                status.submittedAt(),
                Instant.now());

        storeStatus(jobId, cancelledStatus);

        log.info("Job cancelled: jobId={}", jobId);
        return true;
    }

    /**
     * Background worker that processes jobs from queue.
     */
    private void processJobs() {
        while (!Thread.currentThread().isInterrupted()) {
            try {
                // Poll job from queue (blocking)
                AsyncJob job = jobQueue.poll(1, TimeUnit.SECONDS);

                if (job == null) {
                    continue;
                }

                log.info("Processing async job: jobId={}, requestId={}",
                        job.jobId, job.request.getRequestId());

                // Update status to PROCESSING
                updateStatus(job.jobId, "PROCESSING", null, null);

                try {
                    // Execute inference via orchestrator directly
                    InferenceResponse response = orchestrator
                            .execute(job.request.getModel(), job.request, TenantContext.of(job.request.getRequestId())); // Using
                                                                                                                         // requestId
                                                                                                                         // as
                                                                                                                         // placeholder
                                                                                                                         // if
                                                                                                                         // tenantId
                                                                                                                         // missing

                    // Store result
                    updateStatus(job.jobId, "COMPLETED", response, null);

                    log.info("Async job completed: jobId={}, durationMs={}",
                            job.jobId, response.getDurationMs());

                } catch (Exception e) {
                    log.error("Async job failed: jobId={}", job.jobId, e);
                    updateStatus(job.jobId, "FAILED", null, e.getMessage());
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                log.error("Worker thread error", e);
            }
        }
    }

    /**
     * Store job status in Redis and in-memory cache.
     */
    private void storeStatus(String jobId, AsyncJobStatus status) {
        // Store in Redis
        if (hashCommands != null) {
            try {
                Map<String, String> jobData = new HashMap<>();
                jobData.put("jobId", status.jobId());
                jobData.put("requestId", status.requestId());
                jobData.put("tenantId", status.tenantId());
                jobData.put("status", status.status());
                jobData.put("submittedAt", status.submittedAt().toString());

                if (status.completedAt() != null) {
                    jobData.put("completedAt", status.completedAt().toString());
                }
                if (status.error() != null) {
                    jobData.put("error", status.error());
                }
                if (status.result() != null) {
                    // Store minimal result data
                    jobData.put("resultId", status.result().getRequestId());
                    jobData.put("latencyMs", String.valueOf(status.result().getDurationMs()));
                }

                hashCommands.hset(REDIS_JOB_PREFIX + jobId, jobData);

                // Set TTL
                redisDataSource.key().expire(
                        REDIS_JOB_PREFIX + jobId,
                        Duration.ofHours(JOB_TTL_HOURS));

            } catch (Exception e) {
                log.warn("Failed to store job status in Redis", e);
            }
        }

        // Store in memory as fallback
        statusCache.put(jobId, status);
    }

    /**
     * Update job status.
     */
    private void updateStatus(String jobId, String status, InferenceResponse result, String error) {
        AsyncJobStatus currentStatus = getStatus(jobId);
        if (currentStatus == null) {
            log.warn("Cannot update status for unknown job: {}", jobId);
            return;
        }

        AsyncJobStatus newStatus = new AsyncJobStatus(
                jobId,
                currentStatus.requestId(),
                currentStatus.tenantId(),
                status,
                result,
                error,
                currentStatus.submittedAt(),
                "COMPLETED".equals(status) || "FAILED".equals(status)
                        ? Instant.now()
                        : null);

        storeStatus(jobId, newStatus);
    }

    /**
     * Parse job status from Redis hash.
     */
    private AsyncJobStatus parseJobStatus(Map<String, String> jobData) {
        return new AsyncJobStatus(
                jobData.get("jobId"),
                jobData.get("requestId"),
                jobData.get("tenantId"),
                jobData.get("status"),
                null, // Result not stored in hash (too large)
                jobData.get("error"),
                Instant.parse(jobData.get("submittedAt")),
                jobData.containsKey("completedAt")
                        ? Instant.parse(jobData.get("completedAt"))
                        : null);
    }

    /**
     * Scheduled cleanup of old completed jobs.
     */
    @Scheduled(every = "1h")
    void cleanupOldJobs() {
        log.info("Starting cleanup of old async jobs");

        Instant cutoff = Instant.now().minus(Duration.ofHours(JOB_TTL_HOURS));
        int removed = 0;

        // Cleanup in-memory cache
        Iterator<Map.Entry<String, AsyncJobStatus>> iter = statusCache.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry<String, AsyncJobStatus> entry = iter.next();
            AsyncJobStatus status = entry.getValue();

            if (status.isComplete() &&
                    status.completedAt() != null &&
                    status.completedAt().isBefore(cutoff)) {
                iter.remove();
                removed++;
            }
        }

        log.info("Cleaned up {} old async jobs", removed);
    }

    /**
     * Get queue statistics.
     */
    public QueueStats getQueueStats() {
        return new QueueStats(
                jobQueue.size(),
                statusCache.values().stream()
                        .filter(s -> "PROCESSING".equals(s.status()))
                        .count(),
                statusCache.values().stream()
                        .filter(s -> "PENDING".equals(s.status()))
                        .count());
    }

    // ===== Inner Classes =====

    private record AsyncJob(
            String jobId,
            InferenceRequest request,
            int priority,
            Instant submittedAt) {
    }

    public record QueueStats(
            int queueSize,
            long processingCount,
            long pendingCount) {
    }
}
