package tech.kayys.golek.api.model;

/**
 * Represents memory requirements for a model.
 */
public record MemoryRequirements(
        Long minMemoryMb,
        Long recommendedMemoryMb,
        Long maxMemoryMb) {

    public MemoryRequirements {
        if (minMemoryMb != null && minMemoryMb < 0) {
            throw new IllegalArgumentException("Minimum memory cannot be negative");
        }
        if (recommendedMemoryMb != null && recommendedMemoryMb < 0) {
            throw new IllegalArgumentException("Recommended memory cannot be negative");
        }
        if (maxMemoryMb != null && maxMemoryMb < 0) {
            throw new IllegalArgumentException("Maximum memory cannot be negative");
        }
        if (minMemoryMb != null && recommendedMemoryMb != null && minMemoryMb > recommendedMemoryMb) {
            throw new IllegalArgumentException("Minimum memory cannot exceed recommended memory");
        }
        if (recommendedMemoryMb != null && maxMemoryMb != null && recommendedMemoryMb > maxMemoryMb) {
            throw new IllegalArgumentException("Recommended memory cannot exceed maximum memory");
        }
    }
}