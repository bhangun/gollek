package tech.kayys.golek.model.core;

import java.util.Optional;

import io.quarkus.runtime.configuration.MemorySize;

/**
 * Resource requirements and constraints for model execution
 */
public record ResourceRequirements(
                MemorySize minMemory,
                MemorySize recommendedMemory,
                MemorySize minVRAM,
                Optional<Integer> minCores,
                Optional<DiskSpace> diskSpace) {
}