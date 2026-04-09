package tech.kayys.gollek.runtime.inference.batch;

import java.util.ArrayList;
import java.util.List;

/**
 * vLLM-style continuous batching scheduler.
 * <p>
 * Unlike static batching (which waits for a full batch before executing),
 * continuous batching dynamically adds and removes requests each iteration:
 * <pre>
 *   t0: [A, B]        — 2 requests active
 *   t1: [A, B, C]     — C joins mid-flight
 *   t2: [B, C]        — A finishes, removed
 *   t3: [B, C, D, E]  — D, E join
 * </pre>
 * <p>
 * This maximizes GPU utilization and minimizes latency.
 */
public final class ContinuousBatchScheduler {

    private final RequestQueue queue = new RequestQueue();
    private final List<BatchRequest> active = new ArrayList<>();
    private final int maxBatchSize;
    private final java.util.concurrent.atomic.AtomicLong totalBatchedRequests = new java.util.concurrent.atomic.AtomicLong(0);

    private volatile boolean running = false;

    public ContinuousBatchScheduler(int maxBatchSize) {
        this.maxBatchSize = maxBatchSize;
    }

    /** Submit a new inference request for processing. */
    public void submit(BatchRequest request) {
        queue.submit(request);
    }

    /** Start the scheduler loop (runs until stopped). */
    public void start(BatchExecutor executor) {
        running = true;

        while (running) {
            // 1. Drain new requests into active batch
            int available = maxBatchSize - active.size();
            if (available > 0) {
                List<BatchRequest> drained = queue.drain(available);
                if (!drained.isEmpty()) {
                    active.addAll(drained);
                    totalBatchedRequests.addAndGet(drained.size());
                }
            }

            if (active.isEmpty()) {
                sleep(1);
                continue;
            }

            // 2. Execute one step for all active requests
            executor.executeBatchStep(active);

            // 3. Remove finished requests
            active.removeIf(r -> r.finished);
        }
    }

    /** Stop the scheduler loop. */
    public void stop() {
        running = false;
    }

    /** Whether the scheduler is currently running. */
    public boolean isRunning() {
        return running;
    }

    /** Number of currently active requests. */
    public int activeCount() {
        return active.size();
    }

    /** Alias for {@link #activeCount()} used by legacy tests. */
    public int getBatchCount() {
        return activeCount();
    }

    /** Number of pending requests in queue. */
    public int pendingCount() {
        return queue.pendingCount();
    }

    /** Alias for {@link #pendingCount()} used by legacy tests. */
    public int getQueueDepth() {
        return pendingCount();
    }

    /** Total number of requests ever batched by this scheduler. */
    public long getTotalBatchedRequests() {
        return totalBatchedRequests.get();
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Callback interface for executing a single decode step on a batch.
     * Implementors perform the actual model forward pass and token sampling.
     */
    @FunctionalInterface
    public interface BatchExecutor {
        void executeBatchStep(List<BatchRequest> batch);
    }
}
