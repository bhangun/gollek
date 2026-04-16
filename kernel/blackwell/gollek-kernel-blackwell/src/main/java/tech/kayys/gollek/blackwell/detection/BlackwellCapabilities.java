package tech.kayys.gollek.blackwell.detection;

/**
 * Immutable snapshot of NVIDIA Blackwell device capabilities.
 *
 * <p>
 * Produced by {@link BlackwellDetector#detect()} and consumed by
 * {@link tech.kayys.gollek.blackwell.runner.BlackwellRunner#initialize}
 * and Gollek's {@link tech.kayys.gollek.engine.routing.policy.SelectionPolicy}
 * to decide whether to route requests to the Blackwell backend.
 *
 * <p>
 * Blackwell (B100, B200, GB200) introduces:
 * <ul>
 *   <li>TMEM (Tensor Memory) - on-chip accumulator for FlashAttention-3</li>
 *   <li>FP4 tensor cores - 2x throughput over FP8</li>
 *   <li>Async execution engines - concurrent copy/compute</li>
 *   <li>192 GB HBM3e - largest unified memory pool</li>
 * </ul>
 *
 * @param available          true if Blackwell is usable on this host
 * @param cudaComputeCap     compute capability (100 for Blackwell)
 * @param deviceName         friendly name, e.g. "NVIDIA B200"
 * @param gpuCores           number of CUDA cores
 * @param totalMemoryBytes   total GPU memory in bytes
 * @param freeMemoryBytes    available GPU memory in bytes
 * @param tensorCores        number of 4th-gen tensor cores
 * @param tmemSize           TMEM capacity in bytes (typically 64MB)
 * @param smCount            number of streaming multiprocessors
 * @param reason             null if available, otherwise human-readable reason
 */
public record BlackwellCapabilities(
        boolean available,
        int cudaComputeCap,
        String deviceName,
        int gpuCores,
        long totalMemoryBytes,
        long freeMemoryBytes,
        int tensorCores,
        long tmemSize,
        int smCount,
        String reason) {

    /** Convenience factory for unavailable Blackwell. */
    public static BlackwellCapabilities unavailable(String reason) {
        return new BlackwellCapabilities(false, 0, "N/A", 0, 0L, 0L, 0, 0L, 0, reason);
    }

    /**
     * Total memory in gigabytes (rounded to one decimal).
     */
    public double totalMemoryGb() {
        return Math.round(totalMemoryBytes / 1e8) / 10.0;
    }

    /**
     * Free memory in gigabytes (rounded to one decimal).
     */
    public double freeMemoryGb() {
        return Math.round(freeMemoryBytes / 1e8) / 10.0;
    }

    /**
     * TMEM size in megabytes.
     */
    public double tmemMb() {
        return tmemSize / (1024.0 * 1024.0);
    }

    /**
     * Whether this device is Blackwell (compute cap ≥ 10.0).
     */
    public boolean isBlackwell() {
        return available && cudaComputeCap >= 100;
    }

    /**
     * Whether this device supports FlashAttention-3 with TMEM.
     * True for Blackwell (compute capability ≥ 10.0).
     */
    public boolean supportsFlashAttention3() {
        return available && cudaComputeCap >= 100;
    }

    /**
     * Whether this device supports FP4 tensor operations.
     * True for Blackwell B200/GB200.
     */
    public boolean supportsFp4() {
        return available && cudaComputeCap >= 100;
    }

    /**
     * Whether this device supports async execution (concurrent copy/compute).
     */
    public boolean supportsAsyncExecution() {
        return available && cudaComputeCap >= 100;
    }

    /**
     * Whether this device is suitable for very large model inference.
     * True when total memory ≥ 80 GB (B100/B200).
     */
    public boolean isLargeModelCapable() {
        return available && totalMemoryBytes >= 80L * 1024 * 1024 * 1024;
    }

    @Override
    public String toString() {
        if (!available)
            return "BlackwellCapabilities{unavailable: " + reason + "}";
        return String.format(
                "BlackwellCapabilities{device=%s, computeCap=%d.%d, memory=%.1fGB/%.1fGB, TMEM=%.1fMB, SMs=%d, tensorCores=%d}",
                deviceName,
                cudaComputeCap / 10,
                cudaComputeCap % 10,
                freeMemoryGb(),
                totalMemoryGb(),
                tmemMb(),
                smCount,
                tensorCores);
    }
}
