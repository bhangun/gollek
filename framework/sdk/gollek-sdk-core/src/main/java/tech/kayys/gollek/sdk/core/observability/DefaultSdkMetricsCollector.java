package tech.kayys.gollek.sdk.core.observability;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.OperatingSystemMXBean;

/**
 * Default SDK metrics collector that reads directly from JVM MXBeans.
 * Suitable for embedded/local SDK usage without a Micrometer registry.
 *
 * <p>For server-side usage where Micrometer is present, prefer wiring in
 * the {@code ResourceMetrics} bean from {@code gollek-observability} instead.
 */
public class DefaultSdkMetricsCollector implements SdkMetricsCollector {

    private final OperatingSystemMXBean osMxBean;
    private final MemoryMXBean memoryMxBean;

    /** Per-thread snapshot: [cpuTimeNs, heapUsedBytes] */
    private final ThreadLocal<long[]> snapshot = new ThreadLocal<>();

    public DefaultSdkMetricsCollector() {
        this.osMxBean     = ManagementFactory.getOperatingSystemMXBean();
        this.memoryMxBean = ManagementFactory.getMemoryMXBean();
    }

    // ── Inference metrics (no-op by default — override to push to backend) ──

    @Override
    public void recordClientSuccess(String model, long durationMs, int outputTokens) {}

    @Override
    public void recordClientFailure(String model, String errorType) {}

    @Override
    public void recordClientTtft(String model, long ttftMs) {}

    @Override
    public void recordClientTpot(String model, long tpotMs) {}

    @Override
    public void recordClientChunk(String model) {}

    // ── Per-request resource tracking ──────────────────────────────────────

    @Override
    public void onRequestStart(String model) {
        snapshot.set(new long[]{ processCpuTimeNs(), heapUsedNow() });
    }

    @Override
    public void onRequestEnd(String model, long cpuTimeMs, long heapDeltaBytes) {
        snapshot.remove();
        // Override to push these values to Prometheus, Datadog, etc.
    }

    // ── System resource snapshot ───────────────────────────────────────────

    @Override
    public ResourceSnapshot getResourceSnapshot() {
        return new ResourceSnapshot(
                processCpuLoad(),
                systemCpuLoad(),
                memoryMxBean.getHeapMemoryUsage().getUsed(),
                memoryMxBean.getHeapMemoryUsage().getMax(),
                memoryMxBean.getNonHeapMemoryUsage().getUsed(),
                heapUtilization()
        );
    }

    // ── Helper API for subclasses ──────────────────────────────────────────

    protected double processCpuLoad() {
        if (osMxBean instanceof com.sun.management.OperatingSystemMXBean ext) {
            return ext.getProcessCpuLoad();
        }
        return -1.0;
    }

    protected double systemCpuLoad() {
        if (osMxBean instanceof com.sun.management.OperatingSystemMXBean ext) {
            return ext.getCpuLoad();
        }
        return osMxBean.getSystemLoadAverage() / Math.max(1, osMxBean.getAvailableProcessors());
    }

    protected long processCpuTimeNs() {
        if (osMxBean instanceof com.sun.management.OperatingSystemMXBean ext) {
            long t = ext.getProcessCpuTime();
            return t >= 0 ? t : 0;
        }
        return 0;
    }

    protected long heapUsedNow() {
        return memoryMxBean.getHeapMemoryUsage().getUsed();
    }

    protected double heapUtilization() {
        long max = memoryMxBean.getHeapMemoryUsage().getMax();
        return max <= 0 ? 0.0 : (double) heapUsedNow() / max;
    }

    /**
     * Compute per-request resource deltas from the snapshot taken at
     * {@link #onRequestStart}. Returns null if no snapshot is available.
     *
     * @return long[2] = {cpuDeltaNs, heapDeltaBytes} or null
     */
    protected long[] computeRequestDeltas() {
        long[] before = snapshot.get();
        if (before == null) return null;
        return new long[]{
            Math.max(0, processCpuTimeNs() - before[0]),
            heapUsedNow() - before[1]
        };
    }
}
