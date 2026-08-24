package tech.kayys.gollek.metrics;

import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.OperatingSystemMXBean;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Records JVM and OS-level resource metrics for capacity planning.
 *
 * <h3>Metrics Exposed</h3>
 * <ul>
 *   <li>{@code gollek.resource.cpu.process.load}  – CPU usage of the JVM process (0.0–1.0).</li>
 *   <li>{@code gollek.resource.cpu.system.load}   – System-wide CPU load (0.0–1.0).</li>
 *   <li>{@code gollek.resource.memory.heap.used}  – JVM heap used in bytes.</li>
 *   <li>{@code gollek.resource.memory.heap.max}   – JVM heap max in bytes.</li>
 *   <li>{@code gollek.resource.memory.nonheap.used} – Off-heap memory used.</li>
 *   <li>{@code gollek.resource.memory.native.used} – Native/OS RSS (best-effort).</li>
 *   <li>{@code gollek.inference.request.cpu.seconds} – CPU time consumed per inference request.</li>
 *   <li>{@code gollek.inference.request.memory.delta} – Heap delta (bytes) per inference request.</li>
 * </ul>
 *
 * <p>Per-request CPU and memory measurements use a snapshot-diff approach:
 * call {@link #snapshotBefore()} before inference, then {@link #recordRequestResources}
 * after to emit the delta as a histogram.
 */
@ApplicationScoped
public class ResourceMetrics {

    private final MeterRegistry registry;
    private final OperatingSystemMXBean osMxBean;
    private final MemoryMXBean memoryMxBean;

    /** Per-request pending snapshot (thread-local for concurrent requests). */
    private final ThreadLocal<long[]> perRequestSnapshot = new ThreadLocal<>();

    // Atomic gauges for live system readings
    private final AtomicLong heapUsed   = new AtomicLong(0);
    private final AtomicLong heapMax    = new AtomicLong(0);
    private final AtomicLong nonHeapUsed = new AtomicLong(0);

    @Inject
    public ResourceMetrics(MeterRegistry registry) {
        this.registry     = registry;
        this.osMxBean     = ManagementFactory.getOperatingSystemMXBean();
        this.memoryMxBean = ManagementFactory.getMemoryMXBean();
        registerGauges();
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // Gauge registration
    // ──────────────────────────────────────────────────────────────────────────────

    private void registerGauges() {
        // CPU gauges
        Gauge.builder("gollek.resource.cpu.process.load", this, rm -> rm.processCpuLoad())
                .description("CPU load of this JVM process (0.0–1.0)")
                .tag("service", "gollek")
                .register(registry);

        Gauge.builder("gollek.resource.cpu.system.load", this, rm -> rm.systemCpuLoad())
                .description("System-wide CPU load average (0.0–1.0)")
                .tag("service", "gollek")
                .register(registry);

        // Memory gauges – read lazily via AtomicLong updated on demand
        Gauge.builder("gollek.resource.memory.heap.used", heapUsed, AtomicLong::doubleValue)
                .description("JVM heap used in bytes")
                .baseUnit("bytes")
                .tag("service", "gollek")
                .register(registry);

        Gauge.builder("gollek.resource.memory.heap.max", heapMax, AtomicLong::doubleValue)
                .description("JVM heap max in bytes")
                .baseUnit("bytes")
                .tag("service", "gollek")
                .register(registry);

        Gauge.builder("gollek.resource.memory.nonheap.used", nonHeapUsed, AtomicLong::doubleValue)
                .description("JVM off-heap (Metaspace + code cache) used in bytes")
                .baseUnit("bytes")
                .tag("service", "gollek")
                .register(registry);

        // Kick off a lightweight background poller for gauge freshness
        Thread.ofVirtual().name("gollek-resource-poller").start(this::pollLoop);
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // Background polling (virtual thread, 5-second interval)
    // ──────────────────────────────────────────────────────────────────────────────

    private void pollLoop() {
        while (!Thread.currentThread().isInterrupted()) {
            try {
                refreshGauges();
                Thread.sleep(5_000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    private void refreshGauges() {
        heapUsed.set(memoryMxBean.getHeapMemoryUsage().getUsed());
        heapMax.set(memoryMxBean.getHeapMemoryUsage().getMax());
        nonHeapUsed.set(memoryMxBean.getNonHeapMemoryUsage().getUsed());
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // Per-request resource tracking
    // ──────────────────────────────────────────────────────────────────────────────

    /**
     * Take a resource snapshot before an inference request begins.
     * Must be paired with {@link #recordRequestResources(String, String)}.
     */
    public void snapshotBefore() {
        long cpuTimeNs  = processCpuTimeNs();
        long heapBytes  = memoryMxBean.getHeapMemoryUsage().getUsed();
        perRequestSnapshot.set(new long[]{ cpuTimeNs, heapBytes });
    }

    /**
     * Record per-request resource consumption relative to the last
     * {@link #snapshotBefore()} call on the current thread.
     *
     * @param model    model name (tag)
     * @param tenantId tenant/org identifier (tag)
     */
    public void recordRequestResources(String model, String tenantId) {
        long[] before = perRequestSnapshot.get();
        perRequestSnapshot.remove();
        if (before == null) return;

        long cpuDeltaNs  = Math.max(0, processCpuTimeNs() - before[0]);
        long heapDeltaBytes = memoryMxBean.getHeapMemoryUsage().getUsed() - before[1];

        // CPU time histogram (seconds)
        Timer.builder("gollek.inference.request.cpu.seconds")
                .description("CPU time consumed per inference request")
                .tag("model", model)
                .tag("tenant", tenantId)
                .publishPercentiles(0.5, 0.9, 0.99)
                .register(registry)
                .record(Duration.ofNanos(cpuDeltaNs));

        // Heap delta histogram (bytes) – signed; GC might make it negative
        io.micrometer.core.instrument.DistributionSummary
                .builder("gollek.inference.request.memory.delta")
                .description("Heap allocation delta per inference request (bytes)")
                .tag("model", model)
                .tag("tenant", tenantId)
                .baseUnit("bytes")
                .publishPercentiles(0.5, 0.9, 0.99)
                .register(registry)
                .record(Math.max(0, heapDeltaBytes));
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // Snapshot helpers — public for SDK exposure
    // ──────────────────────────────────────────────────────────────────────────────

    /** Returns the CPU load of this JVM process (0.0–1.0, or -1 if unavailable). */
    public double processCpuLoad() {
        if (osMxBean instanceof com.sun.management.OperatingSystemMXBean extended) {
            return extended.getProcessCpuLoad();
        }
        return -1.0;
    }

    /** Returns the system-wide CPU load (0.0–1.0, or -1 if unavailable). */
    public double systemCpuLoad() {
        if (osMxBean instanceof com.sun.management.OperatingSystemMXBean extended) {
            return extended.getCpuLoad();
        }
        return osMxBean.getSystemLoadAverage() / osMxBean.getAvailableProcessors();
    }

    /** Returns JVM heap used in bytes. */
    public long heapUsedBytes() {
        return memoryMxBean.getHeapMemoryUsage().getUsed();
    }

    /** Returns JVM heap max in bytes. */
    public long heapMaxBytes() {
        return memoryMxBean.getHeapMemoryUsage().getMax();
    }

    /** Returns JVM non-heap (Metaspace + code cache) used in bytes. */
    public long nonHeapUsedBytes() {
        return memoryMxBean.getNonHeapMemoryUsage().getUsed();
    }

    /** Returns the heap utilization ratio (0.0–1.0). */
    public double heapUtilization() {
        long max = heapMaxBytes();
        return max <= 0 ? 0.0 : (double) heapUsedBytes() / max;
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ──────────────────────────────────────────────────────────────────────────────

    private long processCpuTimeNs() {
        if (osMxBean instanceof com.sun.management.OperatingSystemMXBean ext) {
            long t = ext.getProcessCpuTime();
            return t >= 0 ? t : 0;
        }
        return 0;
    }
}
