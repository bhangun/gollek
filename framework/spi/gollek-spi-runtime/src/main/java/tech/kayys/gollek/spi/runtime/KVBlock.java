/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.lang.foreign.MemorySegment;

/**
 * Atomic unit of the KV cache for a single transformer layer.
 * Uses off-heap MemorySegments for zero-copy sharing between runners.
 */
public record KVBlock(
    int layerId,
    int headCount,
    int headDim,
    int seqLength,
    MemorySegment keys,
    MemorySegment values,
    DataType dtype
) {
    /**
     * Enum for KV data types.
     */
    public enum DataType {
        FP16,   // 2 bytes
        INT8,   // 1 byte
        Q4_0,   // Quaintized 4-bit (GGML style)
        BF16    // Bfloat16
    }
}
