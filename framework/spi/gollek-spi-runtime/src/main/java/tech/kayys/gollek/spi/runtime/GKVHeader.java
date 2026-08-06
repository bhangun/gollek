/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.nio.ByteOrder;

/**
 * Binary header for the GKV (Gollek KV) format.
 * Matches the C-struct definition for zero-copy mmap interoperability.
 * 64-byte aligned to match CPU cache lines.
 */
public record GKVHeader(
    int magic,              // "GKV1" (0x474B5631)
    int version,            // Format version
    int numLayers,
    int numHeads,
    int headDim,
    int seqLen,
    int dtypeId,            // Map to GKVDataType.id()
    int quantizationId,     // Map to QuantizationTier.id()
    long layerDirOffset,    // Start of layer directory
    long kvDataOffset,      // Start of raw tensor data
    long totalSize,         // Total size of the segment
    ByteOrder endianness    // Runtime endianness (derived from loader)
) {
    public static final int MAGIC = 0x474B5631;
    public static final int HEADER_SIZE = 64; 
}
