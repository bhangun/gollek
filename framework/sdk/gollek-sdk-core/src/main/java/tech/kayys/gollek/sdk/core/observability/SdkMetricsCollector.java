package tech.kayys.gollek.sdk.core.observability;

/**
 * SDK-level metrics interface for tracking inference performance and
 * resource consumption per request.
 *
 * <p>Allows embedding clients (Wayang, custom apps) to instrument their
 * own observability stack independently of the inference server.
 *
 * <h3>Capacity Planning Metrics</h3>
 * <ul>
 *   <li>CPU load (process + system) — for compute sizing</li>
 *   <li>Heap and non-heap memory — for memory capacity planning</li>
 *   <li>Per-request CPU time and heap delta — for per-request cost attribution</li>
 * </ul>
 */
public interface SdkMetricsCollector {

    // ── Inference metrics ──────────────────────────────────────────────────

    void recordClientSuccess(String model, long durationMs, int outputTokens);
    void recordClientFailure(String model, String errorType);

    void recordClientTtft(String model, long ttftMs);
    void recordClientTpot(String model, long tpotMs);
    void recordClientChunk(String model);

    // ── Resource metrics — per-request ─────────────────────────────────────

    /**
     * Called immediately before an inference call begins.
     * Records a baseline CPU and memory snapshot for per-request delta.
     */
    default void onRequestStart(String model) {}

    /**
     * Called immediately after an inference call completes (success or failure).
     * Implementations should emit per-request CPU time and heap delta metrics.
     *
     * @param model           the model name
     * @param cpuTimeMs       approximate CPU time consumed (ms), or -1 if unavailable
     * @param heapDeltaBytes  heap allocation delta (bytes), or -1 if unavailable
     */
    default void onRequestEnd(String model, long cpuTimeMs, long heapDeltaBytes) {}

    // ── System resource snapshot ───────────────────────────────────────────

    /**
     * Returns the current system resource snapshot.
     * Implementations should populate all fields where possible.
     */
    default ResourceSnapshot getResourceSnapshot() {
        return ResourceSnapshot.unavailable();
    }

    // ── Resource snapshot DTO ─────────────────────────────────────────────

    /**
     * Immutable snapshot of current resource utilization.
     * Used for capacity planning and auto-scaling decisions.
     *
     * @param processCpuLoad    CPU load of this process (0.0–1.0, or -1 if unavailable)
     * @param systemCpuLoad     System-wide CPU load (0.0–1.0, or -1 if unavailable)
     * @param heapUsedBytes     JVM heap used in bytes
     * @param heapMaxBytes      JVM heap max in bytes
     * @param nonHeapUsedBytes  JVM off-heap (Metaspace) used in bytes
     * @param heapUtilization   Heap utilization ratio (0.0–1.0)
     */
    record ResourceSnapshot(
            double processCpuLoad,
            double systemCpuLoad,
            long heapUsedBytes,
            long heapMaxBytes,
            long nonHeapUsedBytes,
            double heapUtilization
    ) {
        /** Returns true if resource data is actually available. */
        public boolean isAvailable() {
            return processCpuLoad >= 0;
        }

        /** Returns a snapshot indicating data is not available. */
        public static ResourceSnapshot unavailable() {
            return new ResourceSnapshot(-1, -1, -1, -1, -1, -1);
        }

        /** Returns a human-readable summary for logging. */
        public String summary() {
            if (!isAvailable()) return "[resource metrics unavailable]";
            return String.format(
                    "cpu=%.1f%% (sys=%.1f%%) heap=%dMB/%dMB (%.1f%%) metaspace=%dMB",
                    processCpuLoad * 100, systemCpuLoad * 100,
                    heapUsedBytes / (1024 * 1024), heapMaxBytes / (1024 * 1024),
                    heapUtilization * 100,
                    nonHeapUsedBytes / (1024 * 1024));
        }
    }
}
