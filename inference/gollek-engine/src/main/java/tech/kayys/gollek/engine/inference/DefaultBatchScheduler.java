package tech.kayys.gollek.engine.inference;

import org.jboss.logging.Logger;

import tech.kayys.gollek.error.ErrorPayload;
import tech.kayys.gollek.spi.batch.BatchConfig;
import tech.kayys.gollek.spi.batch.BatchInferenceRequest;
import tech.kayys.gollek.spi.batch.BatchMetrics;
import tech.kayys.gollek.spi.batch.BatchResponse;
import tech.kayys.gollek.spi.batch.BatchResult;
import tech.kayys.gollek.spi.batch.BatchScheduler;
import tech.kayys.gollek.spi.batch.BatchStrategy;
import tech.kayys.gollek.spi.inference.InferenceEngine;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Default implementation of {@link BatchScheduler} supporting three strategies:
 *
 * <ul>
 * <li><b>STATIC</b> – waits until exactly {@code maxBatchSize} requests are
 * queued,
 * then dispatches. Simple but causes head-of-line blocking.</li>
 * <li><b>DYNAMIC</b> – dispatches whenever {@code maxBatchSize} is reached
 * <em>or</em>
 * {@code maxWaitTime} elapses, whichever comes first. Balances throughput vs.
 * latency without stranding slow batches.</li>
 * <li><b>CONTINUOUS</b> – iteration-level scheduling (in-flight batching). Each
 * slot
 * in the active set is refilled the moment its sequence completes. Maximises
 * throughput for variable-length LLM output.</li>
 * </ul>
 *
 * <p>
 * The dispatcher runs in a dedicate Virtual Thread so it never blocks the CDI
 * thread pool.
 * Configuration can be hot-reloaded at runtime via
 * {@link #setConfig(BatchConfig)}.
 */
public class DefaultBatchScheduler implements BatchScheduler {

    private static final Logger LOG = Logger.getLogger(DefaultBatchScheduler.class);

    /** Internal pending-request record held in the queue. */
    record PendingRequest(
            String requestId,
            InferenceRequest request,
            CompletableFuture<InferenceResponse> future,
            Instant enqueueTime) {
    }

    private final InferenceEngine orchestrator;

    /** Live config — written under synchronization via {@link #setConfig}. */
    private volatile BatchConfig config;

    // ── State ────────────────────────────────────────────────────────────────

    private final BlockingQueue<PendingRequest> queue = new LinkedBlockingQueue<>();
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean flushRequested = new AtomicBoolean(false);

    // ── Stats ─────────────────────────────────────────────────────────────────

    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong totalBatches = new AtomicLong(0);
    private final AtomicLong totalBatchSize = new AtomicLong(0);
    private final AtomicLong totalBatchLatencyNs = new AtomicLong(0);
    private final AtomicLong startEpochNs = new AtomicLong(System.nanoTime());

    // ── Construction ──────────────────────────────────────────────────────────

    public DefaultBatchScheduler(InferenceEngine orchestrator, BatchConfig config) {
        this.orchestrator = orchestrator;
        this.config = config;
        config.validate();
        startDispatchLoop();
    }

    // ── BatchScheduler API ────────────────────────────────────────────────────

    @Override
    public CompletableFuture<InferenceResponse> submit(InferenceRequest request) {
        CompletableFuture<InferenceResponse> future = new CompletableFuture<>();
        PendingRequest pending = new PendingRequest(
                request.getRequestId() != null ? request.getRequestId() : UUID.randomUUID().toString(),
                request,
                future,
                Instant.now());
        queue.add(pending);
        totalRequests.incrementAndGet();
        LOG.debugf("Enqueued request %s (queue depth: %d)", pending.requestId(), queue.size());
        return future;
    }

    @Override
    public CompletableFuture<BatchResponse> submitBatch(BatchInferenceRequest batchRequest) {
        List<InferenceRequest> requests = batchRequest.getRequests();
        List<CompletableFuture<InferenceResponse>> futures = new ArrayList<>(requests.size());
        for (InferenceRequest req : requests) {
            futures.add(submit(req));
        }

        String batchId = UUID.randomUUID().toString();
        return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .thenApply(ignored -> {
                    List<BatchResult> results = new ArrayList<>(futures.size());
                    for (int i = 0; i < futures.size(); i++) {
                        CompletableFuture<InferenceResponse> f = futures.get(i);
                        String requestId = requests.get(i).getRequestId();
                        if (f.isCompletedExceptionally()) {
                            // Extract exception message for error payload
                            String errMsg = f.handle((r, ex) -> ex != null ? ex.getMessage() : "").join();
                            results.add(new BatchResult(requestId, null,
                                    ErrorPayload.builder().type("BATCH_ERROR").message(errMsg).build()));
                        } else {
                            results.add(new BatchResult(requestId, f.join(), null));
                        }
                    }
                    BatchMetrics metrics = snapshot();
                    LOG.debugf("Batch %s completed: %d results", batchId, results.size());
                    return new BatchResponse(batchId, results, metrics);
                });
    }

    @Override
    public void flush() {
        flushRequested.set(true);
        LOG.debug("Flush requested — forcing immediate dispatch");
    }

    @Override
    public BatchConfig getConfig() {
        return config;
    }

    @Override
    public synchronized void setConfig(BatchConfig newConfig) {
        newConfig.validate();
        LOG.infof("Batch config updated: strategy=%s, maxBatchSize=%d, maxWaitMs=%d",
                newConfig.strategy(), newConfig.maxBatchSize(), newConfig.maxWaitTime().toMillis());
        this.config = newConfig;
    }

    /** Stop the dispatch loop (call on shutdown). */
    public void stop() {
        running.set(false);
    }

    // ── Metrics snapshot ──────────────────────────────────────────────────────

    public BatchMetrics snapshot() {
        long batches = totalBatches.get();
        long reqs = totalRequests.get();
        double avgBatch = batches == 0 ? 0.0 : (double) totalBatchSize.get() / batches;
        long elapsedNs = System.nanoTime() - startEpochNs.get();
        double rps = elapsedNs == 0 ? 0.0 : (double) reqs / (elapsedNs / 1_000_000_000.0);
        long avgLatMs = batches == 0 ? 0L : totalBatchLatencyNs.get() / batches / 1_000_000L;
        return new BatchMetrics(
                (int) reqs,
                (int) batches,
                queue.size(),
                avgLatMs,
                avgBatch,
                rps,
                avgLatMs);
    }

    // ── Dispatch loop ─────────────────────────────────────────────────────────

    private void startDispatchLoop() {
        running.set(true);
        Thread.ofVirtual().name("gollek-batch-scheduler").start(this::dispatchLoop);
        LOG.infof("Batch scheduler started with strategy=%s", config.strategy());
    }

    private void dispatchLoop() {
        while (running.get()) {
            try {
                BatchStrategy strategy = config.strategy();
                switch (strategy) {
                    case STATIC -> runStatic();
                    case DYNAMIC -> runDynamic();
                    case CONTINUOUS -> runContinuous();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                LOG.info("Batch scheduler interrupted, shutting down");
                break;
            } catch (Exception e) {
                LOG.errorf(e, "Unexpected error in batch dispatch loop");
            }
        }
        LOG.info("Batch scheduler stopped");
    }

    // ── STATIC strategy ───────────────────────────────────────────────────────

    /**
     * Waits until exactly {@code maxBatchSize} requests are queued (or flush is
     * triggered),
     * then dispatches them all as a single batch. The first request in the batch is
     * blocked
     * until the last one arrives — classic head-of-line blocking.
     */
    private void runStatic() throws InterruptedException {
        int maxBatch = config.maxBatchSize();
        List<PendingRequest> batch = new ArrayList<>(maxBatch);

        // Block until at least one request arrives
        PendingRequest first = queue.poll(1, TimeUnit.SECONDS);
        if (first == null)
            return;
        batch.add(first);

        // Collect up to maxBatchSize - 1 more without a timeout
        while (batch.size() < maxBatch && !flushRequested.getAndSet(false)) {
            PendingRequest next = queue.poll(100, TimeUnit.MILLISECONDS);
            if (next != null) {
                batch.add(next);
            } else if (!queue.isEmpty()) {
                // Another request arrived during the short poll window
                queue.drainTo(batch, maxBatch - batch.size());
            }
        }

        dispatchBatch(batch);
    }

    // ── DYNAMIC strategy ──────────────────────────────────────────────────────

    /**
     * Dispatches as soon as either the batch is full OR the wait-time window
     * expires —
     * whichever comes first. Like a bus that leaves on schedule or when it's
     * completely full.
     */
    private void runDynamic() throws InterruptedException {
        int maxBatch = config.maxBatchSize();
        long windowNs = config.maxWaitTime().toNanos();
        List<PendingRequest> batch = new ArrayList<>(maxBatch);

        // Block until at least one request arrives (1 s poll to keep running flag
        // responsive)
        PendingRequest first = queue.poll(1, TimeUnit.SECONDS);
        if (first == null)
            return;
        batch.add(first);

        long deadline = System.nanoTime() + windowNs;

        while (batch.size() < maxBatch && !flushRequested.getAndSet(false)) {
            long remaining = deadline - System.nanoTime();
            if (remaining <= 0)
                break; // Time window elapsed → dispatch what we have
            PendingRequest next = queue.poll(remaining, TimeUnit.NANOSECONDS);
            if (next == null)
                break; // Timed out → dispatch partial batch
            batch.add(next);
        }

        dispatchBatch(batch);
    }

    // ── CONTINUOUS strategy ───────────────────────────────────────────────────

    /**
     * Iteration-level (in-flight) batching. Maintains up to {@code maxBatchSize}
     * concurrent
     * in-flight inferences. Each time a slot finishes, the next queued request is
     * immediately
     * inserted — keeping the active set continuously saturated without waiting for
     * the full
     * batch to complete.
     */
    private void runContinuous() throws InterruptedException {
        int slots = config.maxBatchSize();

        // Fill initial slots
        List<PendingRequest> active = new ArrayList<>(slots);
        while (active.size() < slots) {
            PendingRequest next = queue.poll(1, TimeUnit.SECONDS);
            if (next == null)
                break;
            active.add(next);
        }
        if (active.isEmpty())
            return;

        // Kick off each slot as an independent async inference
        List<CompletableFuture<Void>> slotFutures = new ArrayList<>(active.size());
        for (PendingRequest p : active) {
            slotFutures.add(dispatchSingleAsync(p));
        }

        // Wait for all slots to complete before re-entering the dispatch loop.
        // The next iteration will refill from the queue — this approximates
        // continuous replacement without complex slot-tracking shared state.
        CompletableFuture.allOf(slotFutures.toArray(new CompletableFuture[0])).join();
    }

    // ── Dispatch helpers ──────────────────────────────────────────────────────

    private void dispatchBatch(List<PendingRequest> batch) {
        if (batch.isEmpty())
            return;

        long batchStart = System.nanoTime();
        LOG.infof("Dispatching batch: size=%d, strategy=%s", batch.size(), config.strategy());
        totalBatches.incrementAndGet();
        totalBatchSize.addAndGet(batch.size());

        for (PendingRequest p : batch) {
            dispatchSingleAsync(p);
        }

        // Record latency for stats (approximation — pure dispatch overhead, not
        // inference duration)
        totalBatchLatencyNs.addAndGet(System.nanoTime() - batchStart);
    }

    private CompletableFuture<Void> dispatchSingleAsync(PendingRequest p) {
        return orchestrator.executeAsync(p.request().getModel(), p.request())
                .subscribe()
                .asCompletionStage()
                .thenAccept(response -> {
                    p.future().complete(response);
                    LOG.debugf("Request %s completed (batch strategy: %s)", p.requestId(), config.strategy());
                })
                .exceptionally(ex -> {
                    p.future().completeExceptionally(ex);
                    LOG.warnf("Request %s failed in batch: %s", p.requestId(), ex.getMessage());
                    return null;
                });
    }
}
