package tech.kayys.gollek.inference.gguf;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.Counter;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Handles metrics collection and reporting for GGUF inference operations.
 * Tracks request durations, token counts, coalescing statistics, and performance metrics.
 */
public class LlamaCppMetricsRecorder {

    private MeterRegistry meterRegistry;
    private Tags runnerTags;
    private boolean coalesceMetricsRegistered;

    private final AtomicLong coalesceDrops = new AtomicLong();
    private final AtomicLong coalesceBatches = new AtomicLong();
    private final AtomicLong coalesceBatchTotal = new AtomicLong();
    private final AtomicLong coalesceTasks = new AtomicLong();
    private final AtomicLong coalesceBatchMax = new AtomicLong();
    private final AtomicLong coalesceSeqMaxObserved = new AtomicLong();
    private final AtomicLong coalesceSeqTotal = new AtomicLong();

    public LlamaCppMetricsRecorder() {
        this.coalesceMetricsRegistered = false;
    }

    /**
     * Register metrics with the meter registry.
     */
    public void registerMetrics(MeterRegistry registry, String tenantId, String modelId, int coalesceMaxQueue) {
        if (registry == null) {
            return;
        }
        this.meterRegistry = registry;
        this.runnerTags = Tags.of("provider", "gguf",
                "tenant", tenantId == null ? "unknown" : tenantId,
                "model", modelId == null ? "unknown" : modelId);

        if (coalesceMetricsRegistered) {
            return;
        }

        Tags tags = this.runnerTags;
        registry.gauge("gollek.gguf.coalesce.queue.depth", tags, coalesceMaxQueue, q -> q == null ? 0 : q);
        registry.gauge("gollek.gguf.coalesce.batch.max", tags, coalesceBatchMax, AtomicLong::get);
        registry.gauge("gollek.gguf.coalesce.batches.total", tags, coalesceBatches, AtomicLong::get);
        registry.gauge("gollek.gguf.coalesce.tasks.total", tags, coalesceTasks, AtomicLong::get);
        registry.gauge("gollek.gguf.coalesce.dropped", tags, coalesceDrops, AtomicLong::get);
        registry.gauge("gollek.gguf.coalesce.seq.max", tags, coalesceSeqMaxObserved, AtomicLong::get);
        registry.gauge("gollek.gguf.coalesce.seq.total", tags, coalesceSeqTotal, AtomicLong::get);

        Gauge.builder("gollek.gguf.coalesce.batch.avg", () -> {
            long count = coalesceBatches.get();
            return (count == 0) ? 0.0 : (double) coalesceBatchTotal.get() / count;
        }).tags(tags).register(registry);

        Gauge.builder("gollek.gguf.coalesce.seq.avg", () -> {
            long batches = coalesceBatches.get();
            return (batches == 0) ? 0.0 : (double) coalesceSeqTotal.get() / batches;
        }).tags(tags).register(registry);

        coalesceMetricsRegistered = true;
    }

    /**
     * Record metrics for a completed inference request.
     */
    public void recordInferenceMetrics(
            long requestStartNanos,
            long promptStartNanos,
            long promptEndNanos,
            long decodeStartNanos,
            long firstTokenNanos,
            int inputTokens,
            int outputTokens) {

        if (meterRegistry == null || runnerTags == null) {
            return;
        }

        long requestEnd = System.nanoTime();
        long effectivePromptEnd = promptEndNanos > 0 ? promptEndNanos : requestEnd;
        long effectiveDecodeStart = decodeStartNanos > 0 ? decodeStartNanos : effectivePromptEnd;

        long promptDuration = Math.max(0L, effectivePromptEnd - promptStartNanos);
        long decodeDuration = Math.max(0L, requestEnd - effectiveDecodeStart);
        long requestDuration = Math.max(0L, requestEnd - requestStartNanos);

        Timer.builder("gollek.gguf.request.duration")
                .tags(runnerTags)
                .register(meterRegistry)
                .record(Duration.ofNanos(requestDuration));

        Timer.builder("gollek.gguf.prompt.duration")
                .tags(runnerTags)
                .register(meterRegistry)
                .record(Duration.ofNanos(promptDuration));

        Timer.builder("gollek.gguf.decode.duration")
                .tags(runnerTags)
                .register(meterRegistry)
                .record(Duration.ofNanos(decodeDuration));

        if (firstTokenNanos > 0) {
            long ttft = Math.max(0L, firstTokenNanos - requestStartNanos);
            Timer.builder("gollek.gguf.ttft")
                    .tags(runnerTags)
                    .register(meterRegistry)
                    .record(Duration.ofNanos(ttft));
        }

        if (outputTokens > 0 && decodeDuration > 0) {
            long tpot = Math.max(1L, decodeDuration / outputTokens);
            Timer.builder("gollek.gguf.tpot")
                    .tags(runnerTags)
                    .register(meterRegistry)
                    .record(Duration.ofNanos(tpot));
        }

        meterRegistry.counter("gollek.gguf.tokens.input", runnerTags)
                .increment(Math.max(0, inputTokens));
        meterRegistry.counter("gollek.gguf.tokens.output", runnerTags)
                .increment(Math.max(0, outputTokens));
    }

    /**
     * Record coalescing batch statistics.
     */
    public void recordCoalesceBatch(int batchSize) {
        coalesceBatches.incrementAndGet();
        coalesceBatchTotal.addAndGet(batchSize);
        coalesceTasks.addAndGet(batchSize);
        updateCoalesceBatchMax(batchSize);
    }

    /**
     * Record coalescing sequence statistics.
     */
    public void recordCoalesceSequences(int seqCount) {
        if (seqCount <= 0) {
            return;
        }
        coalesceSeqTotal.addAndGet(seqCount);
        long previous;
        do {
            previous = coalesceSeqMaxObserved.get();
            if (seqCount <= previous) {
                return;
            }
        } while (!coalesceSeqMaxObserved.compareAndSet(previous, seqCount));
    }

    /**
     * Record a dropped coalescing request.
     */
    public void recordCoalesceDrop() {
        coalesceDrops.incrementAndGet();
    }

    /**
     * Get the coalesce drops counter.
     */
    public AtomicLong getCoalesceDrops() {
        return coalesceDrops;
    }

    /**
     * Get the coalesce batches counter.
     */
    public AtomicLong getCoalesceBatches() {
        return coalesceBatches;
    }

    /**
     * Get the coalesce batch total counter.
     */
    public AtomicLong getCoalesceBatchTotal() {
        return coalesceBatchTotal;
    }

    /**
     * Get the coalesce tasks counter.
     */
    public AtomicLong getCoalesceTasks() {
        return coalesceTasks;
    }

    /**
     * Get the coalesce batch max gauge.
     */
    public AtomicLong getCoalesceBatchMax() {
        return coalesceBatchMax;
    }

    /**
     * Get the coalesce sequence max observed gauge.
     */
    public AtomicLong getCoalesceSeqMaxObserved() {
        return coalesceSeqMaxObserved;
    }

    /**
     * Get the coalesce sequence total counter.
     */
    public AtomicLong getCoalesceSeqTotal() {
        return coalesceSeqTotal;
    }

    private void updateCoalesceBatchMax(int size) {
        long previous;
        do {
            previous = coalesceBatchMax.get();
            if (size <= previous) {
                return;
            }
        } while (!coalesceBatchMax.compareAndSet(previous, size));
    }

    /**
     * Reset all metrics (useful for testing).
     */
    public void reset() {
        coalesceDrops.set(0);
        coalesceBatches.set(0);
        coalesceBatchTotal.set(0);
        coalesceTasks.set(0);
        coalesceBatchMax.set(0);
        coalesceSeqMaxObserved.set(0);
        coalesceSeqTotal.set(0);
        coalesceMetricsRegistered = false;
        meterRegistry = null;
        runnerTags = null;
    }
}
