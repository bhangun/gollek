package tech.kayys.gollek.metal.detection;

/**
 * Immutable snapshot of Apple Silicon / Metal capabilities on this host.
 *
 * <p>
 * Produced by {@link AppleSiliconDetector#detect()} and consumed by
 * {@link tech.kayys.gollek.extension.metal.runner.MetalRunner#initialize}
 * and Gollek's {@link tech.kayys.gollek.engine.routing.policy.SelectionPolicy}
 * to decide whether to route requests to the Metal backend.
 *
 * @param available          true if Metal is usable on this host
 * @param appleSilicon       true if the CPU is Apple Silicon (arm64)
 * @param chipName           friendly name, e.g. "Apple M3 Pro"
 * @param gpuCores           number of GPU cores (0 if unknown)
 * @param unifiedMemoryBytes total shared DRAM in bytes
 * @param unifiedMemory      true on Apple Silicon (CPU+GPU share DRAM)
 * @param reason             null if available, otherwise human-readable reason
 */
public record MetalCapabilities(
        boolean available,
        boolean appleSilicon,
        String chipName,
        int gpuCores,
        long unifiedMemoryBytes,
        boolean unifiedMemory,
        String reason) {
    /** Convenience factory for unavailable Metal. */
    public static MetalCapabilities unavailable(String reason) {
        return new MetalCapabilities(false, false, "N/A", 0, 0L, false, reason);
    }

    /**
     * Unified memory in gigabytes (rounded to one decimal).
     * Returns 0.0 if unifiedMemoryBytes is unknown.
     */
    public double unifiedMemoryGb() {
        return Math.round(unifiedMemoryBytes / 1e8) / 10.0;
    }

    /**
     * Whether this device is suitable for large-model inference on a single chip.
     * True when unified memory ≥ 16 GB (M1 Pro / Max or better).
     */
    public boolean isLargeModelCapable() {
        return available && unifiedMemoryBytes >= 16L * 1024 * 1024 * 1024;
    }

    @Override
    public String toString() {
        if (!available)
            return "MetalCapabilities{unavailable: " + reason + "}";
        return String.format(
                "MetalCapabilities{chip=%s, gpuCores=%d, unifiedMemory=%.1fGB, appleSilicon=%s}",
                chipName, gpuCores, unifiedMemoryGb(), appleSilicon);
    }
}
