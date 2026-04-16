package tech.kayys.gollek.inference.libtorch;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.observability.AdapterMetricSchema;
import tech.kayys.gollek.spi.observability.AdapterMetricsRecorder;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Micrometer-based metrics for the LibTorch provider.
 * <p>
 * Tracks request counts, success/failure rates, inference durations,
 * and token throughput — matching the observability level of the GGUF provider.
 */
@ApplicationScoped
public class LibTorchMetrics {

    private static final Logger log = Logger.getLogger(LibTorchMetrics.class);
    private static final String PREFIX = "libtorch.";
    private static final String PROVIDER_TAG = "libtorch";

    private Counter requestCounter;
    private Counter successCounter;
    private Counter failureCounter;
    private Counter tokensGeneratedCounter;
    private Counter adapterPairCacheHitCounter;
    private Counter adapterPairCacheMissCounter;
    private Counter advancedHybridAttemptCounter;
    private Counter advancedHybridSuccessCounter;
    private Counter advancedHybridFallbackCounter;
    private Counter fp8RowwiseDecisionCounter;
    private Timer inferenceTimer;
    private Timer adapterInitWaitTimer;
    private Timer adapterApplyTimer;
    private Counter sessionEvictionReclaimedCounter;

    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong totalSuccess = new AtomicLong(0);
    private final AtomicLong totalFailure = new AtomicLong(0);
    private final AtomicLong totalTokens = new AtomicLong(0);
    private final AtomicLong adapterPairCacheHits = new AtomicLong(0);
    private final AtomicLong adapterPairCacheMisses = new AtomicLong(0);
    private final AtomicLong sessionEvictionIdleTimeoutSeconds = new AtomicLong(300);
    private final AtomicLong sessionEvictionPressurePermille = new AtomicLong(0);
    private MeterRegistry meterRegistry;

    @Inject
    AdapterMetricsRecorder adapterMetricsRecorder;

    @Inject
    void init(MeterRegistry registry) {
        log.debug("Init : " + registry.toString());
        this.meterRegistry = registry;
        this.requestCounter = Counter.builder(PREFIX + "requests")
                .description("Total inference requests")
                .register(registry);
        this.successCounter = Counter.builder(PREFIX + "success")
                .description("Successful inference requests")
                .register(registry);
        this.failureCounter = Counter.builder(PREFIX + "failure")
                .description("Failed inference requests")
                .register(registry);
        this.tokensGeneratedCounter = Counter.builder(PREFIX + "tokens.generated")
                .description("Total tokens generated")
                .register(registry);
        this.inferenceTimer = Timer.builder(PREFIX + "infer.duration")
                .description("Inference request duration")
                .register(registry);
        this.adapterInitWaitTimer = Timer.builder(PREFIX + "adapter.init.wait.duration")
                .description("Time spent waiting for adapter initialization lock")
                .register(registry);
        this.adapterApplyTimer = Timer.builder(PREFIX + "adapter.apply.duration")
                .description("Time spent applying runtime adapter weights to a session")
                .register(registry);
        this.adapterPairCacheHitCounter = Counter.builder(PREFIX + "adapter.pair_cache.hit")
                .description("Adapter pair-index cache hits")
                .register(registry);
        this.adapterPairCacheMissCounter = Counter.builder(PREFIX + "adapter.pair_cache.miss")
                .description("Adapter pair-index cache misses")
                .register(registry);
        this.advancedHybridAttemptCounter = Counter.builder(PREFIX + "advanced.hybrid.attempt")
                .description("Hybrid FP8/BF16 attention attempts")
                .register(registry);
        this.advancedHybridSuccessCounter = Counter.builder(PREFIX + "advanced.hybrid.success")
                .description("Hybrid FP8/BF16 attention successful executions")
                .register(registry);
        this.advancedHybridFallbackCounter = Counter.builder(PREFIX + "advanced.hybrid.fallback")
                .description("Hybrid FP8/BF16 attention fallbacks to baseline")
                .register(registry);
        this.fp8RowwiseDecisionCounter = Counter.builder(PREFIX + "advanced.fp8.rowwise.decision")
                .description("FP8 rowwise planner decisions")
                .register(registry);
        this.sessionEvictionReclaimedCounter = Counter.builder("gollek.session.eviction.reclaimed_total")
                .tag("provider", PROVIDER_TAG)
                .description("Total sessions reclaimed by idle-eviction loops")
                .register(registry);

        // GPU memory gauges
        io.micrometer.core.instrument.Gauge.builder(PREFIX + "gpu.memory.allocated", this::getGpuMemoryAllocated)
                .description("GPU memory currently allocated (bytes)")
                .baseUnit("bytes")
                .register(registry);
        io.micrometer.core.instrument.Gauge.builder(PREFIX + "gpu.memory.reserved", this::getGpuMemoryReserved)
                .description("GPU memory reserved by caching allocator (bytes)")
                .baseUnit("bytes")
                .register(registry);
        io.micrometer.core.instrument.Gauge.builder(PREFIX + "gpu.memory.max", this::getGpuMemoryMax)
                .description("Peak GPU memory allocated (bytes)")
                .baseUnit("bytes")
                .register(registry);
        io.micrometer.core.instrument.Gauge.builder("gollek.session.eviction.idle_timeout.seconds",
                sessionEvictionIdleTimeoutSeconds, AtomicLong::get)
                .tag("provider", PROVIDER_TAG)
                .description("Adaptive idle-timeout selected by session eviction policy")
                .baseUnit("seconds")
                .register(registry);
        io.micrometer.core.instrument.Gauge.builder("gollek.session.eviction.pressure.score",
                sessionEvictionPressurePermille, value -> value.get() / 1000.0d)
                .tag("provider", PROVIDER_TAG)
                .description("Adaptive session-eviction pressure score (0..1)")
                .register(registry);
    }

    public void recordRequest() {
        totalRequests.incrementAndGet();
        if (requestCounter != null)
            requestCounter.increment();
    }

    public void recordSuccess() {
        totalSuccess.incrementAndGet();
        if (successCounter != null)
            successCounter.increment();
    }

    public void recordFailure() {
        totalFailure.incrementAndGet();
        if (failureCounter != null)
            failureCounter.increment();
    }

    public void recordDuration(Duration duration) {
        if (inferenceTimer != null)
            inferenceTimer.record(duration);
    }

    public void recordTokensGenerated(long count) {
        totalTokens.addAndGet(count);
        if (tokensGeneratedCounter != null)
            tokensGeneratedCounter.increment(count);
    }

    public void recordInference(String modelId, boolean success, Duration duration) {
        recordRequest();
        if (success) {
            recordSuccess();
        } else {
            recordFailure();
        }
        recordDuration(duration);
    }

    public void recordAdapterInitWait(Duration duration) {
        if (duration == null) {
            return;
        }
        if (adapterInitWaitTimer != null) {
            adapterInitWaitTimer.record(duration);
        }
    }

    public void recordAdapterApply(Duration duration) {
        if (duration == null) {
            return;
        }
        if (adapterApplyTimer != null) {
            adapterApplyTimer.record(duration);
        }
    }

    public void recordAdapterPairCacheHit() {
        adapterPairCacheHits.incrementAndGet();
        if (adapterPairCacheHitCounter != null) {
            adapterPairCacheHitCounter.increment();
        }
    }

    public void recordAdapterPairCacheMiss() {
        adapterPairCacheMisses.incrementAndGet();
        if (adapterPairCacheMissCounter != null) {
            adapterPairCacheMissCounter.increment();
        }
    }

    public void recordSessionEvictionTelemetry(double pressureScore, int idleTimeoutSeconds, int reclaimedSessions) {
        sessionEvictionPressurePermille.set(Math.round((float) (pressureScore * 1000.0d)));
        sessionEvictionIdleTimeoutSeconds.set(Math.max(1, idleTimeoutSeconds));
        if (reclaimedSessions > 0 && sessionEvictionReclaimedCounter != null) {
            sessionEvictionReclaimedCounter.increment(reclaimedSessions);
        }
    }

    public void recordAdvancedModeResolution(LibTorchAdvancedModeResolver.EffectiveAdvancedMode mode) {
        if (mode == null) {
            return;
        }
        if (meterRegistry == null) {
            return;
        }
        try {
            meterRegistry.counter(
                    PREFIX + "advanced.mode.resolution.total",
                    "enabled", Boolean.toString(mode.advancedEnabled()),
                    "attention_mode", mode.attentionMode(),
                    "reason", mode.reason(),
                    "detected_sm", mode.detectedGpuSm().map(Object::toString).orElse("unknown"))
                    .increment();
        } catch (Throwable ignored) {
            // best-effort metric
        }
    }

    public void recordAdvancedHybridAttempt() {
        if (advancedHybridAttemptCounter != null) {
            advancedHybridAttemptCounter.increment();
        }
    }

    public void recordAdvancedHybridSuccess() {
        if (advancedHybridSuccessCounter != null) {
            advancedHybridSuccessCounter.increment();
        }
    }

    public void recordAdvancedHybridFallback() {
        if (advancedHybridFallbackCounter != null) {
            advancedHybridFallbackCounter.increment();
        }
    }

    public void recordFp8RowwiseDecision(boolean enabled, String reason) {
        if (meterRegistry == null) {
            return;
        }
        meterRegistry.counter(
                PREFIX + "advanced.fp8.rowwise.decision.total",
                "enabled", Boolean.toString(enabled),
                "reason", reason == null || reason.isBlank() ? "unknown" : reason)
                .increment();
        if (fp8RowwiseDecisionCounter != null) {
            fp8RowwiseDecisionCounter.increment();
        }
    }

    // ── GPU memory tracking ─────────────────────────────────────────

    /**
     * Get current GPU memory allocated (bytes).
     * Returns 0 if GPU is not available.
     */
    public double getGpuMemoryAllocated() {
        try {
            var binding = tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding.getInstance();
            if (binding.hasSymbol("at_cuda_memory_allocated")) {
                var fn = binding.bind("at_cuda_memory_allocated",
                        java.lang.foreign.FunctionDescriptor.of(java.lang.foreign.ValueLayout.JAVA_LONG,
                                java.lang.foreign.ValueLayout.JAVA_INT));
                return (double) (long) fn.invoke(0);
            }
        } catch (Throwable ignored) {
        }
        return 0.0;
    }

    /**
     * Get GPU memory reserved by caching allocator (bytes).
     */
    public double getGpuMemoryReserved() {
        try {
            var binding = tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding.getInstance();
            if (binding.hasSymbol("at_cuda_memory_reserved")) {
                var fn = binding.bind("at_cuda_memory_reserved",
                        java.lang.foreign.FunctionDescriptor.of(java.lang.foreign.ValueLayout.JAVA_LONG,
                                java.lang.foreign.ValueLayout.JAVA_INT));
                return (double) (long) fn.invoke(0);
            }
        } catch (Throwable ignored) {
        }
        return 0.0;
    }

    /**
     * Get peak GPU memory allocated (bytes).
     */
    public double getGpuMemoryMax() {
        try {
            var binding = tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding.getInstance();
            if (binding.hasSymbol("at_cuda_max_memory_allocated")) {
                var fn = binding.bind("at_cuda_max_memory_allocated",
                        java.lang.foreign.FunctionDescriptor.of(java.lang.foreign.ValueLayout.JAVA_LONG,
                                java.lang.foreign.ValueLayout.JAVA_INT));
                return (double) (long) fn.invoke(0);
            }
        } catch (Throwable ignored) {
        }
        return 0.0;
    }

    // ── Accessor methods ────────────────────────────────────────────

    public long getTotalRequests() {
        return totalRequests.get();
    }

    public long getTotalSuccess() {
        return totalSuccess.get();
    }

    public long getTotalFailure() {
        return totalFailure.get();
    }

    public long getTotalTokens() {
        return totalTokens.get();
    }

    public long getAdapterPairCacheHits() {
        return adapterPairCacheHits.get();
    }

    public long getAdapterPairCacheMisses() {
        return adapterPairCacheMisses.get();
    }
}
