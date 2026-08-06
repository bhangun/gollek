/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import jakarta.enterprise.context.ApplicationScoped;

/**
 * Service registry to manage and resolve attention kernels at runtime.
 * Allows the platform to switch between Flash, Paged, or Standard kernels
 * based on hardware capabilities and context length.
 */
@ApplicationScoped
public class AttentionKernelRegistry {

    private final Map<KernelType, AttentionKernel> kernels = new ConcurrentHashMap<>();

    /**
     * Registers a new attention kernel implementation.
     */
    public void register(AttentionKernel kernel) {
        kernels.put(kernel.type(), kernel);
    }

    /**
     * Resolves the best available kernel for a specific type.
     */
    public Optional<AttentionKernel> resolve(KernelType type) {
        return Optional.ofNullable(kernels.get(type));
    }

    /**
     * Resolves the optimal kernel based on hardware and context.
     * (Placeholder for more complex logic: e.g. if GPU found and seqLen > 1024 -> FLASH)
     */
    public AttentionKernel resolveOptimal(KernelType preferredType) {
        return resolve(preferredType).orElseGet(() -> 
            resolve(KernelType.STANDARD).orElseThrow(() -> 
                new IllegalStateException("No attention kernels registered!")));
    }
}
