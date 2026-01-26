package tech.kayys.golek.api.model;

/**
 * Represents the resource requirements for running a model.
 */
public record ResourceRequirements(
                MemoryRequirements memory,
                ComputeRequirements compute,
                StorageRequirements storage) {
}