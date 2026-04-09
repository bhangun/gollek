package tech.kayys.gollek.runtime.inference.batch;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.PriorityBlockingQueue;

/**
 * Thread-safe queue for incoming inference requests.
 * <p>
 * This queue maintains {@link BatchRequest}s in priority order.
 * The {@link ContinuousBatchScheduler} drains this queue each iteration
 * to dynamically add new requests to the active batch.
 */
public final class RequestQueue {

    private final PriorityBlockingQueue<BatchRequest> queue = new PriorityBlockingQueue<>();

    /** Submit a new inference request. */
    public void submit(BatchRequest req) {
        queue.add(req);
    }

    /**
     * Drain up to {@code maxBatchSize} requests from the queue in priority order.
     *
     * @param maxBatchSize maximum number of requests to drain
     * @return list of drained requests (may be empty)
     */
    public List<BatchRequest> drain(int maxBatchSize) {
        List<BatchRequest> batch = new ArrayList<>();

        while (batch.size() < maxBatchSize) {
            BatchRequest r = queue.poll();
            if (r == null) break;
            batch.add(r);
        }

        return batch;
    }

    /** Number of pending requests. */
    public int pendingCount() {
        return queue.size();
    }

    /** Whether the queue is empty. */
    public boolean isEmpty() {
        return queue.isEmpty();
    }
}
