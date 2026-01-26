package tech.kayys.golek.api.model;

/**
 * Represents storage requirements for a model.
 */
public record StorageRequirements(
        Long minStorageMb,
        Long recommendedStorageMb,
        Long maxStorageMb) {

    public StorageRequirements {
        if (minStorageMb != null && minStorageMb < 0) {
            throw new IllegalArgumentException("Minimum storage cannot be negative");
        }
        if (recommendedStorageMb != null && recommendedStorageMb < 0) {
            throw new IllegalArgumentException("Recommended storage cannot be negative");
        }
        if (maxStorageMb != null && maxStorageMb < 0) {
            throw new IllegalArgumentException("Maximum storage cannot be negative");
        }
        if (minStorageMb != null && recommendedStorageMb != null && minStorageMb > recommendedStorageMb) {
            throw new IllegalArgumentException("Minimum storage cannot exceed recommended storage");
        }
        if (recommendedStorageMb != null && maxStorageMb != null && recommendedStorageMb > maxStorageMb) {
            throw new IllegalArgumentException("Recommended storage cannot exceed maximum storage");
        }
    }
}