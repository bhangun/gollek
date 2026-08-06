/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Utility for GKV memory alignment and structure calculations.
 * Ensures compatibility with CPU cache lines (64 bytes) and GPU memory (256 bytes).
 */
public class GKVMemoryUtils {

    public static final int CPU_ALIGNMENT = 64;
    public static final int GPU_ALIGNMENT = 256;

    /**
     * Aligns an offset to the specified power-of-two alignment.
     */
    public static long align(long offset, int alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    /**
     * Calculates the required size for a GKV binary segment.
     */
    public static long calculateRequiredSize(int numLayers, int numHeads, int headDim, int seqLen, GKVDataType dtype) {
        long size = GKVHeader.HEADER_SIZE;
        size = align(size, CPU_ALIGNMENT);
        
        // Layer directory
        long dirSize = (long) numLayers * GKVLayerEntry.ENTRY_SIZE;
        size += align(dirSize, CPU_ALIGNMENT);
        
        // KV Data
        long layerSize = (long) seqLen * numHeads * headDim * dtype.sizeBytes();
        long totalKvSize = layerSize * numLayers * 2; // K + V
        
        return size + align(totalKvSize, GPU_ALIGNMENT);
    }
}
