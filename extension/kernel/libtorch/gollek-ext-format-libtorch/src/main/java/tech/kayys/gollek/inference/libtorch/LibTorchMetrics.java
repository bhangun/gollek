package tech.kayys.gollek.inference.libtorch;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

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

    private Counter requestCounter;
    private Counter successCounter;
    private Counter failureCounter;
    private Counter tokensGeneratedCounter;
    private Timer inferenceTimer;

    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong totalSuccess = new AtomicLong(0);
    private final AtomicLong totalFailure = new AtomicLong(0);
    private final AtomicLong totalTokens = new AtomicLong(0);

    @Inject
    void init(MeterRegistry registry) {
        log.debug("Init : " + registry.toString());
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
}
