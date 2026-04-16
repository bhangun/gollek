package tech.kayys.gollek.waitscheduler;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import tech.kayys.gollek.waitscheduler.binding.ForelenBinding;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.spi.batch.BatchConfig;
import tech.kayys.gollek.spi.batch.BatchScheduler;
import tech.kayys.gollek.spi.batch.BatchStrategy;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;


import java.time.Duration;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Fluid Inference Scheduler — WAIT KV-capacity-aware admission control.
 *
 * <p>Wraps Gollek's {@link BatchScheduler} with a KV-capacity admission gate
 * driven by live pool occupancy from {@link PagedKVCacheManager} and output-
 * length predictions from {@link ForelenBinding}.
 *
 * <h2>Papers</h2>
 * <ul>
 *   <li>WAIT (arXiv:2504.11320, January 2026) — fluid admission threshold</li>
 *   <li>ForeLen (ICLR 2026) — lightweight output-length predictor</li>
 * </ul>
 *
 * <h2>WAIT admission condition</h2>
 * <pre>
 *   admit iff: current_kv_blocks + E[blocks(r)] ≤ total_kv_blocks × kvCapacityFraction
 * </pre>
 * {@code E[blocks(r)]} comes from {@link ForelenBinding#predict(int)} (~0.5 ms CPU),
 * falling back to a heuristic when the native library is absent.
 *
 * <h3>Config</h3>
 * <pre>
 *   gollek.scheduler.fluid.enabled=false
 *   gollek.scheduler.fluid.kv-capacity-fraction=0.85
 *   gollek.scheduler.fluid.forelen-library=/opt/gollek/lib/libgollek_forelen.so
 *   gollek.scheduler.fluid.homogeneous-batching=true
 *   gollek.scheduler.fluid.max-queue-depth=4096
 *   gollek.scheduler.fluid.admission-poll-ms=2
 *   gollek.scheduler.fluid.scan-depth=64
 * </pre>
 *
 * <h3>REST</h3>
 * <pre>
 *   GET  /v1/scheduler/stats
 *   PUT  /v1/scheduler/threshold?fraction=0.9
 *   GET  /v1/scheduler/simulate?promptTokens=4096
 * </pre>
 */
@ApplicationScoped
@Path("/v1/scheduler")
@Produces(MediaType.APPLICATION_JSON)
public class FluidInferenceScheduler {

    private static final Logger LOG = Logger.getLogger(FluidInferenceScheduler.class);

    @ConfigProperty(name = "gollek.scheduler.fluid.enabled",              defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.scheduler.fluid.kv-capacity-fraction", defaultValue = "0.85")
    double kvCapacityFraction;

    @ConfigProperty(name = "gollek.scheduler.fluid.forelen-library",
                    defaultValue = "/opt/gollek/lib/libgollek_forelen.so")
    String forelenLibrary;

    @ConfigProperty(name = "gollek.scheduler.fluid.homogeneous-batching", defaultValue = "true")
    boolean homogeneousBatching;

    @ConfigProperty(name = "gollek.scheduler.fluid.max-queue-depth",      defaultValue = "4096")
    int maxQueueDepth;

    @ConfigProperty(name = "gollek.scheduler.fluid.admission-poll-ms",    defaultValue = "2")
    long admissionPollMs;

    @ConfigProperty(name = "gollek.scheduler.fluid.scan-depth",           defaultValue = "64")
    int scanDepth;

    @Inject BatchScheduler        batchScheduler;
    @Inject PagedKVCacheManager   kvCacheManager;

    private ForelenBinding forelenBinding;
    private LinkedBlockingDeque<PendingRequest> waitQueue;
    private volatile Thread admissionThread;

    // Stats
    private final AtomicLong totalAdmitted   = new AtomicLong();
    private final AtomicLong totalQueued     = new AtomicLong();
    private final AtomicLong totalRejected   = new AtomicLong();
    private final AtomicLong totalPredictions = new AtomicLong();
    private final AtomicLong totalBypassed   = new AtomicLong();

    /** Pending request record. */
    private record PendingRequest(
            String requestId,
            InferenceRequest request,
            CompletableFuture<InferenceResponse> future,
            int predictedKvBlocks,
            long enqueueNano) {}

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @PostConstruct
    public void start() {
        if (!enabled) {
            LOG.info("[FluidScheduler] Disabled (gollek.scheduler.fluid.enabled=false)");
            return;
        }

        // Load ForeLen binding — falls back to heuristic if .so absent
        ForelenBinding.initialize(java.nio.file.Path.of(forelenLibrary));
        forelenBinding = ForelenBinding.getInstance();
        LOG.infof("[FluidScheduler] ForeLen: native=%s", forelenBinding.isNativeAvailable());

        // Put underlying BatchScheduler in continuous mode with minimal wait window
        batchScheduler.setConfig(new BatchConfig(
                BatchStrategy.CONTINUOUS, 32, Duration.ofMillis(1), 8, 8, 128, false));

        waitQueue = new LinkedBlockingDeque<>(maxQueueDepth);
        admissionThread = Thread.ofVirtual().name("fluid-admission").start(this::admissionLoop);

        LOG.infof("[FluidScheduler] Started — fraction=%.2f queue=%d poll=%dms",
                kvCapacityFraction, maxQueueDepth, admissionPollMs);
    }

    @PreDestroy
    public void stop() {
        if (admissionThread != null) admissionThread.interrupt();
    }

    // ── Admission API ─────────────────────────────────────────────────────────

    /**
     * Submit a request through the WAIT fluid admission gate.
     *
     * <p>If {@code current_kv_blocks + E[blocks(r)] ≤ capacity × fraction}
     * the request is forwarded to {@link BatchScheduler#submit} immediately.
     * Otherwise it queues until blocks free up.
     */
    public CompletableFuture<InferenceResponse> submit(InferenceRequest request) {
        if (!enabled) return batchScheduler.submit(request);

        int promptTokens = estimatePromptTokens(request);
        int outputTokens = forelenBinding != null
                ? forelenBinding.predict(promptTokens)
                : heuristicPredict(promptTokens);
        int kvBlocks = tokensToBlocks(promptTokens + outputTokens);
        totalPredictions.incrementAndGet();

        CompletableFuture<InferenceResponse> future = new CompletableFuture<>();

        if (admitsNow(kvBlocks)) {
            totalAdmitted.incrementAndGet();
            batchScheduler.submit(request)
                    .thenAccept(future::complete)
                    .exceptionally(ex -> { future.completeExceptionally(ex); return null; });
        } else {
            String reqId = request.getRequestId() != null
                    ? request.getRequestId() : UUID.randomUUID().toString();
            PendingRequest pending = new PendingRequest(
                    reqId, request, future, kvBlocks, System.nanoTime());

            if (!waitQueue.offer(pending)) {
                totalRejected.incrementAndGet();
                future.completeExceptionally(new RejectedExecutionException(
                        "Fluid scheduler queue full (depth=" + waitQueue.size() + ")"));
            } else {
                totalQueued.incrementAndGet();
                LOG.debugf("[FluidScheduler] Queued %s (predicted %d KV blocks)", reqId, kvBlocks);
            }
        }
        return future;
    }

    /**
     * Notify the scheduler that a request completed and KV blocks were freed.
     * Wakes the admission loop to re-check queued requests.
     *
     * @param requestId       completed request ID
     * @param actualOutputLen actual output token count (for ForeLen online update)
     */
    public void complete(String requestId, int actualOutputLen) {
        // Online ForeLen update (no-op if native unavailable)
        if (forelenBinding != null && forelenBinding.isNativeAvailable()) {
            // Would call forelenBinding.update() with actual vs predicted here
        }
        // Wake admission loop
        if (admissionThread != null) admissionThread.interrupt();
    }

    // ── WAIT admission condition ──────────────────────────────────────────────

    private boolean admitsNow(int predictedBlocks) {
        int current  = kvCacheManager.getAllocatedBlockCount();
        int ceiling  = (int)(kvCacheManager.getConfig().getTotalBlocks() * kvCapacityFraction);
        return (current + predictedBlocks) <= ceiling;
    }

    private PendingRequest findAdmissible() {
        PendingRequest head = waitQueue.peekFirst();
        if (head == null) return null;
        if (admitsNow(head.predictedKvBlocks())) return head;
        if (scanDepth <= 1) return null;

        int scanned = 0;
        for (PendingRequest r : waitQueue) {
            if (scanned++ >= scanDepth) break;
            if (r == head) continue;
            if (admitsNow(r.predictedKvBlocks())) return r;
        }
        return null;
    }

    // Visible for tests
    PendingRequest findAdmissibleForTest() {
        return findAdmissible();
    }

    private void admissionLoop() {
        while (!Thread.currentThread().isInterrupted()) {
            try {
                PendingRequest candidate = findAdmissible();
                if (candidate == null) {
                    Thread.sleep(admissionPollMs);
                    continue;
                }
                PendingRequest head = waitQueue.peekFirst();
                PendingRequest r;
                if (candidate == head) {
                    r = waitQueue.pollFirst();
                } else if (waitQueue.remove(candidate)) {
                    r = candidate;
                    totalBypassed.incrementAndGet();
                } else {
                    continue;
                }

                if (r != null) {
                    totalAdmitted.incrementAndGet();
                    batchScheduler.submit(r.request())
                            .thenAccept(r.future()::complete)
                            .exceptionally(ex -> { r.future().completeExceptionally(ex); return null; });
                    LOG.debugf("[FluidScheduler] Admitted queued %s (%d blocks)",
                            r.requestId(), r.predictedKvBlocks());
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt(); // clear and re-check immediately
                Thread.interrupted();
            }
        }
    }

    // ── REST management ───────────────────────────────────────────────────────

    @GET @Path("/stats")
    public Response stats() {
        var s = kvCacheManager.getStats();
        return Response.ok(Map.ofEntries(
                Map.entry("enabled",             enabled),
                Map.entry("queue_depth",         waitQueue != null ? waitQueue.size() : 0),
                Map.entry("total_admitted",      totalAdmitted.get()),
                Map.entry("total_queued",        totalQueued.get()),
                Map.entry("total_rejected",      totalRejected.get()),
                Map.entry("total_predictions",   totalPredictions.get()),
                Map.entry("total_bypassed",      totalBypassed.get()),
                Map.entry("forelen_mae",         forelenBinding != null ? forelenBinding.mae() : Float.NaN),
                Map.entry("kv_utilization",      s.utilization()),
                Map.entry("kv_free_blocks",      s.freeBlocks()),
                Map.entry("capacity_fraction",   kvCapacityFraction),
                Map.entry("scan_depth",          scanDepth)
        )).build();
    }

    @PUT @Path("/threshold")
    public Response setThreshold(@QueryParam("fraction") double fraction) {
        if (fraction <= 0.0 || fraction > 1.0)
            return Response.status(400).entity(Map.of("error", "fraction ∈ (0,1]")).build();
        double old = kvCapacityFraction;
        kvCapacityFraction = fraction;
        if (admissionThread != null) admissionThread.interrupt(); // wake to re-check
        return Response.ok(Map.of("previous", old, "current", fraction)).build();
    }

    @GET @Path("/simulate")
    public Response simulate(@QueryParam("promptTokens") int promptTokens) {
        int outputTokens = forelenBinding != null
                ? forelenBinding.predict(promptTokens)
                : heuristicPredict(promptTokens);
        int kvBlocks   = tokensToBlocks(promptTokens + outputTokens);
        boolean admits = admitsNow(kvBlocks);
        return Response.ok(Map.of(
                "prompt_tokens",          promptTokens,
                "predicted_output_tokens", outputTokens,
                "predicted_kv_blocks",    kvBlocks,
                "would_admit",            admits,
                "current_kv_blocks",      kvCacheManager.getAllocatedBlockCount(),
                "capacity_ceiling",       (int)(kvCacheManager.getConfig().getTotalBlocks()
                                               * kvCapacityFraction)
        )).build();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private int estimatePromptTokens(InferenceRequest request) {
        if (request.getPromptTokenCount() > 0) return request.getPromptTokenCount();
        return request.getMessages().stream()
                .mapToInt(m -> m.getContent() != null ? m.getContent().length() / 4 : 0)
                .sum();
    }

    private int heuristicPredict(int promptTokens) {
        return Math.max(64, Math.min(2048, promptTokens / 2));
    }

    private int tokensToBlocks(int tokens) {
        int blockSize = kvCacheManager.getConfig().getBlockSize();
        return Math.max(1, (tokens + blockSize - 1) / blockSize);
    }
}
