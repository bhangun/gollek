package tech.kayys.golek.spi.model;

/**
 * Represents the resource requirements for running a model.
 */
public record ResourceRequirements(
                MemoryRequirements memory,
                ComputeRequirements compute,
                StorageRequirements storage) {
}