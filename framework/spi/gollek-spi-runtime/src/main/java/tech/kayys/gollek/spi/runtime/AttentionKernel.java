/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.lang.foreign.MemorySegment;

/**
 * Abstraction for the attention computation kernel.
 * Decouples the mathematical attention operation from the inference runner,
 * allowing hardware-specific optimizations (Flash, Metal, etc.) to be plugged in.
 */
public interface AttentionKernel {

    /**
     * Computes the attention operation for a specific query against a KV block.
     *
     * @param kv      the KV memory block (local or remote/mapped)
     * @param query   the query tensor segment
     * @param output  the output tensor segment where results are written
     * @param context additional parameters (scaling, mask, etc.)
     */
    void compute(KVBlock kv, MemorySegment query, MemorySegment output, AttentionContext context);

    /**
     * The type of strategy implemented by this kernel.
     */
    KernelType type();

    /**
     * Additional metadata for the attention operation.
     */
    record AttentionContext(
        float scale,
        int headIndex,
        int seqPos,
        MemorySegment mask // Optional attention mask
    ) {}
}
