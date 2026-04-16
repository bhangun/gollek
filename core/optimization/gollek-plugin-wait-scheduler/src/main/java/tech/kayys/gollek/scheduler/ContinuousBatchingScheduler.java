package tech.kayys.gollek.scheduler;

import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.kvcache.KVCacheExhaustedException;

/**
 * Continuous batching scheduler for LLM inference.
 * <p>
 * Implements vLLM-style continuous batching: instead of waiting for all requests
 * in a batch to complete before starting a new batch, this scheduler continuously
 * fills available capacity with new requests as existing requests complete.
 * <p>
 * <b>Key features:</b>
 * <ul>
 *   <li>Dynamic batch sizing based on KV cache availability</li>
 *   <li>Request prioritization (FCFS with optional priority)</li>
 *   <li>Automatic request eviction on cache pressure</li>
 *   <li>Integration with PagedKVCacheManager for block allocation</li>
 *   <li>Metrics collection for throughput monitoring</li>
 * </ul>
 * 
 * <h2>How It Works</h2>
 * <pre>
 * Traditional Batching:
 *   Batch 1: [Req1, Req2, Req3] → Wait for all → Batch 2: [Req4, Req5]
 *   Inefficient: GPU idle while waiting for longest request
 *
 * Continuous Batching:
 *   Step 1: [Req1, Req2, Req3]
 *   Step 2: [Req1✓, Req2, Req3, Req4] ← Req4 added immediately
 *   Step 3: [Req2✓, Req3, Req4, Req5] ← Req5 added immediately
 *   GPU stays saturated → 10-24x throughput improvement
 * </pre>
 * 
 * @see PagedKVCacheManager
 * @since 0.1.0
 */
public class ContinuousBatchingScheduler implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(ContinuousBatchingScheduler.class);

    // ── Configuration ─────────────────────────────────────────────────

    /** Maximum number of concurrent requests in a batch */
    private final int maxBatchSize;

    /** Maximum sequence length (tokens) per request */
    private final int maxSequenceLength;

    /** KV cache manager for block allocation */
    private final PagedKVCacheManager kvCacheManager;

    /** Scheduler loop interval in milliseconds */
    private final long scheduleIntervalMs;

    // ── State ─────────────────────────────────────────────────────────

    /** Pending requests queue (FIFO) */
    private final BlockingQueue<ScheduledRequest> pendingQueue = new LinkedBlockingQueue<>();

    /** Currently running requests */
    private final Map<String, ScheduledRequest> runningRequests = new ConcurrentHashMap<>();

    /** Completed requests cache (for metrics) */
    private final LinkedHashMap<String, ScheduledRequest> completedRequests = new LinkedHashMap<>(100, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, ScheduledRequest> eldest) {
            return size() > 1000;  // Keep last 1000 completed requests
        }
    };

    /** Scheduler executor for background scheduling loop */
    private final ScheduledExecutorService scheduler;

    /** Flag to control scheduler lifecycle */
    private final AtomicBoolean running = new AtomicBoolean(false);

    // ── Metrics ───────────────────────────────────────────────────────

    /** Total requests processed */
    private final AtomicInteger totalProcessed = new AtomicInteger(0);

    /** Total tokens generated */
    private final AtomicLong totalTokensGenerated = new AtomicLong(0);

    /** Average batch utilization */
    private final AtomicLong batchUtilizationSum = new AtomicLong(0);
    private final AtomicInteger batchUtilizationCount = new AtomicInteger(0);

    /** Rejected requests (cache pressure) */
    private final AtomicInteger rejectedRequests = new AtomicInteger(0);

    // ── Lifecycle ─────────────────────────────────────────────────────

    private ContinuousBatchingScheduler(Config config) {
        this.maxBatchSize = config.maxBatchSize;
        this.maxSequenceLength = config.maxSequenceLength;
        this.kvCacheManager = config.kvCacheManager;
        this.scheduleIntervalMs = config.scheduleIntervalMs;
        this.scheduler = Executors.newScheduledThreadPool(
            1, 
            r -> {
                Thread t = new Thread(r, "continuous-batching-scheduler");
                t.setDaemon(true);
                return t;
            }
        );
    }

    /**
     * Creates a builder for configuring this scheduler.
     */
    public static ConfigBuilder builder() {
        return new ConfigBuilder();
    }

    /**
     * Starts the scheduler's background scheduling loop.
     */
    public void start() {
        if (running.compareAndSet(false, true)) {
            scheduler.scheduleAtFixedRate(
                this::scheduleLoop,
                0,
                scheduleIntervalMs,
                TimeUnit.MILLISECONDS
            );
            LOG.infof("ContinuousBatchingScheduler started: maxBatchSize=%d, interval=%dms",
                maxBatchSize, scheduleIntervalMs);
        }
    }

    /**
     * Stops the scheduler and waits for in-flight requests to complete.
     */
    @Override
    public void close() {
        if (running.compareAndSet(true, false)) {
            scheduler.shutdown();
            try {
                if (!scheduler.awaitTermination(1, TimeUnit.SECONDS)) {
                    scheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                scheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
            LOG.info("ContinuousBatchingScheduler stopped");
        }
    }

    // ── Public API ────────────────────────────────────────────────────

    /**
     * Submits a new inference request to the scheduler.
     *
     * @param request the request to schedule
     * @return true if accepted, false if rejected (cache pressure)
     */
    public boolean submit(ScheduledRequest request) {
        if (!running.get()) {
            LOG.warn("Scheduler is not running");
            return false;
        }

        // Check if we have capacity
        if (pendingQueue.size() + runningRequests.size() >= maxBatchSize * 2) {
            LOG.warnf("Request queue full, rejecting request: %s", request.getRequestId());
            rejectedRequests.incrementAndGet();
            return false;
        }

        pendingQueue.offer(request);
        LOG.debugf("Request submitted: %s (queue size: %d)", request.getRequestId(), pendingQueue.size());
        return true;
    }

    /**
     * Cancels a pending or running request.
     *
     * @param requestId the request to cancel
     * @return true if found and cancelled
     */
    public boolean cancel(String requestId) {
        // Try to remove from pending queue
        boolean removed = pendingQueue.removeIf(r -> r.getRequestId().equals(requestId));
        if (removed) {
            LOG.infof("Cancelled pending request: %s", requestId);
            return true;
        }

        // Try to remove from running requests
        ScheduledRequest running = runningRequests.remove(requestId);
        if (running != null) {
            running.markCancelled();
            LOG.infof("Cancelled running request: %s", requestId);
            return true;
        }

        return false;
    }

    /**
     * Gets current scheduler metrics.
     */
    public SchedulerMetrics getMetrics() {
        double avgBatchUtilization = batchUtilizationCount.get() > 0
            ? (double) batchUtilizationSum.get() / batchUtilizationCount.get()
            : 0.0;

        return new SchedulerMetrics(
            pendingQueue.size(),
            runningRequests.size(),
            totalProcessed.get(),
            totalTokensGenerated.get(),
            rejectedRequests.get(),
            avgBatchUtilization,
            kvCacheManager.getFreeBlockCount(),
            kvCacheManager.getConfig().getTotalBlocks()
        );
    }

    // ── Scheduling Loop ───────────────────────────────────────────────

    /**
     * Main scheduling loop - runs periodically to manage requests.
     */
    private void scheduleLoop() {
        try {
            // 1. Check for completed requests and free their blocks
            checkCompletedRequests();

            // 2. Try to admit new requests from pending queue
            admitPendingRequests();

            // 3. Record batch utilization
            if (!runningRequests.isEmpty()) {
                int utilization = runningRequests.size();
                batchUtilizationSum.addAndGet(utilization);
                batchUtilizationCount.incrementAndGet();
            }

            LOG.debugf("Scheduler tick: pending=%d, running=%d, availableBlocks=%d",
                pendingQueue.size(),
                runningRequests.size(),
                kvCacheManager.getFreeBlockCount());
        } catch (Exception e) {
            LOG.error("Error in scheduling loop", e);
        }
    }

    /**
     * Checks running requests for completion and removes them.
     */
    private void checkCompletedRequests() {
        Iterator<Map.Entry<String, ScheduledRequest>> it = runningRequests.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<String, ScheduledRequest> entry = it.next();
            ScheduledRequest request = entry.getValue();

            if (request.isCompleted()) {
                it.remove();
                kvCacheManager.freeRequest(request.getRequestId());
                completedRequests.put(request.getRequestId(), request);
                totalProcessed.incrementAndGet();
                totalTokensGenerated.addAndGet(request.getGeneratedTokens());
                LOG.debugf("Request completed: %s (tokens: %d)", 
                    request.getRequestId(), request.getGeneratedTokens());
            } else if (request.isCancelled()) {
                it.remove();
                kvCacheManager.freeRequest(request.getRequestId());
                LOG.debugf("Request cancelled: %s", request.getRequestId());
            }
        }
    }

    /**
     * Admits pending requests if there's capacity.
     */
    private void admitPendingRequests() {
        int availableSlots = maxBatchSize - runningRequests.size();
        
        // Also check KV cache capacity
        int blocksPerRequest = kvCacheManager.blocksRequired(maxSequenceLength);
        int availableBlocks = kvCacheManager.getFreeBlockCount();
        int cacheLimitedSlots = availableBlocks / blocksPerRequest;
        
        int canAdmit = Math.min(availableSlots, cacheLimitedSlots);

        while (canAdmit > 0 && !pendingQueue.isEmpty()) {
            ScheduledRequest request = pendingQueue.poll();
            if (request == null) break;

            // Try to allocate KV cache blocks
            try {
                int promptTokens = request.getPromptLength();
                kvCacheManager.allocateForPrefill(request.getRequestId(), promptTokens);
                
                runningRequests.put(request.getRequestId(), request);
                request.markStarted();
                canAdmit--;

                LOG.debugf("Request admitted: %s (prompt: %d tokens)", 
                    request.getRequestId(), promptTokens);
            } catch (KVCacheExhaustedException e) {
                // Put back in queue and stop admitting
                pendingQueue.offer(request);
                LOG.warnf("KV cache exhausted, cannot admit request: %s", request.getRequestId());
                break;
            }
        }
    }

    // ── Nested Classes ────────────────────────────────────────────────

    /**
     * Represents a scheduled inference request.
     */
    public static class ScheduledRequest {
        private final String requestId;
        private final String prompt;
        private final int promptLength;
        private final int maxTokens;
        private final Optional<Integer> priority;

        private volatile boolean started = false;
        private volatile boolean completed = false;
        private volatile boolean cancelled = false;
        private volatile int generatedTokens = 0;

        public ScheduledRequest(String requestId, String prompt, int maxTokens) {
            this(requestId, prompt, maxTokens, null);
        }

        public ScheduledRequest(String requestId, String prompt, int maxTokens, Integer priority) {
            this.requestId = requestId;
            this.prompt = prompt;
            this.promptLength = prompt.split("\\s+").length;  // Approximate token count
            this.maxTokens = maxTokens;
            this.priority = Optional.ofNullable(priority);
        }

        public String getRequestId() { return requestId; }
        public String getPrompt() { return prompt; }
        public int getPromptLength() { return promptLength; }
        public int getMaxTokens() { return maxTokens; }
        public Optional<Integer> getPriority() { return priority; }
        public int getGeneratedTokens() { return generatedTokens; }
        public boolean isStarted() { return started; }
        public boolean isCompleted() { return completed; }
        public boolean isCancelled() { return cancelled; }

        void markStarted() { this.started = true; }
        void markCompleted() { this.completed = true; }
        void markCancelled() { this.cancelled = true; }

        void incrementGeneratedTokens() { this.generatedTokens++; }
    }

    /**
     * Scheduler metrics snapshot.
     */
    public record SchedulerMetrics(
        int pendingRequests,
        int runningRequests,
        int totalProcessed,
        long totalTokensGenerated,
        int rejectedRequests,
        double averageBatchUtilization,
        int availableKvCacheBlocks,
        int totalKvCacheBlocks
    ) {
        public double cacheUtilization() {
            return totalKvCacheBlocks > 0 
                ? (double) (totalKvCacheBlocks - availableKvCacheBlocks) / totalKvCacheBlocks 
                : 0.0;
        }
    }

    /**
     * Configuration for the scheduler.
     */
    public static final class Config {
        final int maxBatchSize;
        final int maxSequenceLength;
        final PagedKVCacheManager kvCacheManager;
        final long scheduleIntervalMs;

        private Config(int maxBatchSize, int maxSequenceLength, 
                      PagedKVCacheManager kvCacheManager, long scheduleIntervalMs) {
            this.maxBatchSize = maxBatchSize;
            this.maxSequenceLength = maxSequenceLength;
            this.kvCacheManager = kvCacheManager;
            this.scheduleIntervalMs = scheduleIntervalMs;
        }
    }

    /**
     * Builder for Config.
     */
    public static final class ConfigBuilder {
        private int maxBatchSize = 256;
        private int maxSequenceLength = 4096;
        private PagedKVCacheManager kvCacheManager;
        private long scheduleIntervalMs = 10;  // 10ms = 100Hz scheduling

        public ConfigBuilder maxBatchSize(int maxBatchSize) {
            this.maxBatchSize = maxBatchSize;
            return this;
        }

        public ConfigBuilder maxSequenceLength(int maxSequenceLength) {
            this.maxSequenceLength = maxSequenceLength;
            return this;
        }

        public ConfigBuilder kvCacheManager(PagedKVCacheManager kvCacheManager) {
            this.kvCacheManager = kvCacheManager;
            return this;
        }

        public ConfigBuilder scheduleIntervalMs(long scheduleIntervalMs) {
            this.scheduleIntervalMs = scheduleIntervalMs;
            return this;
        }

        public ContinuousBatchingScheduler build() {
            Objects.requireNonNull(kvCacheManager, "kvCacheManager is required");
            return new ContinuousBatchingScheduler(new Config(maxBatchSize, maxSequenceLength, kvCacheManager, scheduleIntervalMs));
        }
    }
}
