/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Types of attention kernels supported by Gollek.
 * Defines the computational strategy used for the attention operation.
 */
public enum KernelType {
    STANDARD,      // Baseline (O(n²)) generic attention
    FLASH,         // FlashAttention (memory-optimized fused kernel)
    PAGED,         // PagedAttention (for sharded/distributed KV memory)
    SPARSE,        // Sparse/Sliding-window attention for long contexts
    FUSED,         // Hardware-specific (e.g. METAL/CUDA) fully fused layer
    LITE           // Optimized for mobile or edge devices
}
