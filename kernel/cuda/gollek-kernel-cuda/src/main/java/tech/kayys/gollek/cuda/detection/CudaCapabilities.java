package tech.kayys.gollek.cuda.detection;

/**
 * Immutable snapshot of NVIDIA CUDA device capabilities.
 *
 * <p>
 * Produced by {@link CudaDetector#detect()} and consumed by
 * {@link tech.kayys.gollek.cuda.runner.CudaRunner#initialize}
 * and Gollek's {@link tech.kayys.gollek.engine.routing.policy.SelectionPolicy}
 * to decide whether to route requests to the CUDA backend.
 *
 * @param available          true if CUDA is usable on this host
 * @param cudaComputeCap     compute capability version (e.g., 80 for A100, 90 for H100)
 * @param deviceName         friendly name, e.g. "NVIDIA A100 80GB PCIe"
 * @param gpuCores           number of CUDA cores (0 if unknown)
 * @param totalMemoryBytes   total GPU memory in bytes
 * @param freeMemoryBytes    available GPU memory in bytes
 * @param multiProcessorCount number of streaming multiprocessors
 * @param reason             null if available, otherwise human-readable reason
 */
public record CudaCapabilities(
        boolean available,
        int cudaComputeCap,
        String deviceName,
        int gpuCores,
        long totalMemoryBytes,
        long freeMemoryBytes,
        int multiProcessorCount,
        String reason) {
    
    /** Convenience factory for unavailable CUDA. */
    public static CudaCapabilities unavailable(String reason) {
        return new CudaCapabilities(false, 0, "N/A", 0, 0L, 0L, 0, reason);
    }

    /**
     * Total memory in gigabytes (rounded to one decimal).
     * Returns 0.0 if totalMemoryBytes is unknown.
     */
    public double totalMemoryGb() {
        return Math.round(totalMemoryBytes / 1e8) / 10.0;
    }

    /**
     * Free memory in gigabytes (rounded to one decimal).
     * Returns 0.0 if freeMemoryBytes is unknown.
     */
    public double freeMemoryGb() {
        return Math.round(freeMemoryBytes / 1e8) / 10.0;
    }

    /**
     * Whether this device is suitable for large-model inference.
     * True when total memory ≥ 24 GB (A100/H100 or better).
     */
    public boolean isLargeModelCapable() {
        return available && totalMemoryBytes >= 24L * 1024 * 1024 * 1024;
    }

    /**
     * Whether this device supports FlashAttention-2/3 optimizations.
     * True for compute capability ≥ 8.0 (A100+).
     */
    public boolean supportsFlashAttention() {
        return available && cudaComputeCap >= 80;
    }

    /**
     * Whether this device supports FP8 tensor operations.
     * True for compute capability ≥ 9.0 (H100+).
     */
    public boolean supportsFp8() {
        return available && cudaComputeCap >= 90;
    }

    @Override
    public String toString() {
        if (!available)
            return "CudaCapabilities{unavailable: " + reason + "}";
        return String.format(
                "CudaCapabilities{device=%s, computeCap=%d.%d, memory=%.1fGB/%.1fGB, SMs=%d, cores=%d}",
                deviceName, 
                cudaComputeCap / 10, 
                cudaComputeCap % 10,
                freeMemoryGb(), 
                totalMemoryGb(), 
                multiProcessorCount,
                gpuCores);
    }
}
