package tech.kayys.gollek.spi.spec;

import tech.kayys.alkhawarizm.core.tensor.DeviceType;

/**
 * Represents the resource requirements for running a model.
 */
public record ResourceRequirements(
                MemoryRequirements memory,
                ComputeRequirements compute,
                StorageRequirements storage) {
}