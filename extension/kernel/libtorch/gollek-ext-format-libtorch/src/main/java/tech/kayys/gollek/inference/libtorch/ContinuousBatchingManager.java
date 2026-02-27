package tech.kayys.gollek.inference.libtorch;

import io.quarkus.runtime.StartupEvent;
import io.quarkus.runtime.ShutdownEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.gollek.inference.libtorch.core.Tensor;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Continuous Batching Manager for LibTorch inference.
 * <p>
 * Accumulates individual inference requests and combines them into batches
 * for efficient GPU utilization. Uses a <strong>priority queue</strong> so that
 * higher-priority requests (e.g. interactive chat) are processed before
 * lower-priority ones (e.g. background summarization).
 * <p>
 * Uses the high-level {@link Tensor} API ({@code cat}, {@code indexSelect})
 * for zero-copy tensor operations.
 * <p>
 * The batcher runs in a dedicated Virtual Thread and is conditionally
 * started based on configuration.
 */
@ApplicationScoped
public class ContinuousBatchingManager {
    private static final Logger log = Logger.getLogger(ContinuousBatchingManager.class);

    /**
     * Request priority levels. Lower ordinal = higher priority.
     */
    public enum Priority {
        CRITICAL(0),
        HIGH(1),
        NORMAL(2),
        LOW(3);

        private final int level;

        Priority(int level) {
            this.level = level;
        }

        public int level() {
            return level;
        }

        public static Priority fromString(String s) {
            if (s == null)
                return NORMAL;
            return switch (s.toLowerCase().trim()) {
                case "critical" -> CRITICAL;
                case "high" -> HIGH;
                case "low" -> LOW;
                default -> NORMAL;
            };
        }
    }

    private final PriorityBlockingQueue<BatchRequest> queue = new PriorityBlockingQueue<>();
    private final AtomicLong batchCount = new AtomicLong(0);
    private final AtomicLong totalBatchedRequests = new AtomicLong(0);
    private volatile boolean running = false;

    @Inject
    LibTorchSessionManager sessionManager;

    @Inject
    LibTorchProviderConfig config;

    void onStart(@Observes StartupEvent ev) {
        if (config.batching().enabled()) {
            running = true;
            Thread.ofVirtual().name("gollek-batcher").start(this::batchingLoop);
            log.debugf("Continuous batching started (maxBatch=%d, timeoutMs=%d)",
                    config.batching().maxBatchSize(), config.batching().batchTimeoutMs());
        } else {
            log.debug("Continuous batching is disabled");
        }
    }

    void onShutdown(@Observes ShutdownEvent ev) {
        running = false;
    }

    /**
     * Enqueue a request for batched inference.
     *
     * @param request the batch request
     * @return a future that completes with the inference response
     */
    public CompletableFuture<InferenceResponse> enqueue(BatchRequest request) {
        if (!running) {
            request.future().completeExceptionally(
                    new IllegalStateException("Batching manager is not running"));
            return request.future();
        }
        queue.add(request);
        return request.future();
    }

    public boolean isRunning() {
        return running;
    }

    public long getBatchCount() {
        return batchCount.get();
    }

    public long getTotalBatchedRequests() {
        return totalBatchedRequests.get();
    }

    // ── Batching loop ────────────────────────────────────────────────

    private void batchingLoop() {
        int maxBatchSize = config.batching().maxBatchSize();
        int timeoutMs = config.batching().batchTimeoutMs();

        while (running) {
            try {
                List<BatchRequest> batch = new ArrayList<>();

                // Wait for at least one request
                BatchRequest first = queue.poll(1, TimeUnit.SECONDS);
                if (first == null)
                    continue; // timeout, re-check running flag
                batch.add(first);

                // Drain additional requests up to max batch, with short timeout
                // PriorityBlockingQueue ensures highest-priority items come first
                long deadline = System.nanoTime() + (timeoutMs * 1_000_000L);
                while (batch.size() < maxBatchSize) {
                    long remaining = deadline - System.nanoTime();
                    if (remaining <= 0)
                        break;
                    BatchRequest next = queue.poll(remaining, TimeUnit.NANOSECONDS);
                    if (next == null)
                        break;
                    batch.add(next);
                }

                // Sort batch so higher-priority requests get first tensor slots
                batch.sort(null);

                if (log.isDebugEnabled()) {
                    log.debugf("Processing batch of %d requests (priorities: %s)",
                            batch.size(),
                            batch.stream().map(r -> r.priority().name())
                                    .collect(java.util.stream.Collectors.joining(", ")));
                }

                processBatch(batch);
                batchCount.incrementAndGet();
                totalBatchedRequests.addAndGet(batch.size());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                log.error("Batch processing loop error", e);
            }
        }
        log.debug("Batching loop stopped");
    }

    // ── Batch processing ─────────────────────────────────────────────

    private void processBatch(List<BatchRequest> batch) {
        try {
            // 1. Concatenate all input tensors along dim 0
            List<Tensor> inputs = batch.stream()
                    .map(BatchRequest::input)
                    .toList();
            Tensor batchedInput = Tensor.cat(inputs, 0);

            // 2. Execute forward pass using session from first request
            var first = batch.get(0);
            LibTorchSessionManager.SessionContext session = null;
            try {
                session = sessionManager.getSession(first.tenantId(), first.modelId(), config);
                TorchScriptRunner runner = session.runner();
                Tensor batchedOutput = runner.forward(batchedInput);

                // 3. Slice results and deliver to individual futures
                sliceAndDeliver(batch, batchedOutput);

                // Cleanup batched tensors
                batchedOutput.close();
            } finally {
                if (session != null) {
                    sessionManager.releaseSession(first.tenantId(), first.modelId(), session);
                }
                batchedInput.close();
            }
        } catch (Throwable t) {
            // Complete all futures exceptionally
            batch.forEach(req -> req.future().completeExceptionally(
                    new RuntimeException("Batch inference failed", t)));
        }
    }

    /**
     * Slices the batched output tensor and completes individual request futures.
     */
    private void sliceAndDeliver(List<BatchRequest> batch, Tensor batchedOutput) {
        for (int i = 0; i < batch.size(); i++) {
            BatchRequest req = batch.get(i);
            try {
                // Create a 1-D index tensor [i] for slicing
                try (Tensor indexTensor = Tensor.fromLongArray(new long[] { i }, new long[] { 1 })) {
                    // Slice via index_select on dim 0
                    try (Tensor result = batchedOutput.indexSelect(0, indexTensor)) {
                        // Use the handler provided by the request to process the tensor slice
                        InferenceResponse response = req.handler().apply(result);
                        req.future().complete(response);
                    }
                }
            } catch (Throwable t) {
                req.future().completeExceptionally(
                        new RuntimeException("Failed to slice batch output for request " + req.requestId(), t));
            }
        }
    }

    /**
     * Get the current queue depth (pending requests).
     */
    public int getQueueDepth() {
        return queue.size();
    }

    // ── Request record ───────────────────────────────────────────────

    /**
     * Represents a single inference request queued for batching.
     * Implements {@link Comparable} for priority-based ordering in the queue.
     *
     * @param tenantId   tenant identifier
     * @param modelId    model identifier
     * @param modelPath  path to the model file
     * @param requestId  unique request identifier
     * @param priority   request priority level
     * @param enqueuedAt timestamp when the request was enqueued (for FIFO within
     *                   same priority)
     * @param input      input tensor for this request
     * @param handler    function to process the output tensor slice and return an
     *                   InferenceResponse
     * @param future     completable future for the response
     */
    public record BatchRequest(
            String tenantId, String modelId, Path modelPath, String requestId,
            Priority priority, long enqueuedAt,
            Tensor input, java.util.function.Function<Tensor, InferenceResponse> handler,
            CompletableFuture<InferenceResponse> future) implements Comparable<BatchRequest> {

        public BatchRequest(
                String tenantId, String modelId, Path modelPath, String requestId,
                Tensor input, java.util.function.Function<Tensor, InferenceResponse> handler,
                CompletableFuture<InferenceResponse> future) {
            this(tenantId, modelId, modelPath, requestId, Priority.NORMAL,
                    System.nanoTime(), input, handler, future);
        }

        @Override
        public int compareTo(BatchRequest other) {
            int cmp = Integer.compare(this.priority.level(), other.priority.level());
            if (cmp != 0)
                return cmp;
            return Long.compare(this.enqueuedAt, other.enqueuedAt); // FIFO within same priority
        }
    }
}
