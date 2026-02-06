package tech.kayys.golek.spi.model;

/**
 * Represents compute requirements for a model.
 */
public record ComputeRequirements(
        Integer minCpuCores,
        Double minCpuSpeedGhz,
        Boolean gpuRequired,
        String minGpuMemoryGb) {

    public ComputeRequirements {
        if (minCpuCores != null && minCpuCores <= 0) {
            throw new IllegalArgumentException("Minimum CPU cores must be positive");
        }
        if (minCpuSpeedGhz != null && minCpuSpeedGhz <= 0) {
            throw new IllegalArgumentException("Minimum CPU speed must be positive");
        }
        if (gpuRequired != null && gpuRequired && (minGpuMemoryGb == null || minGpuMemoryGb.isBlank())) {
            throw new IllegalArgumentException("GPU memory requirement must be specified when GPU is required");
        }
    }
}