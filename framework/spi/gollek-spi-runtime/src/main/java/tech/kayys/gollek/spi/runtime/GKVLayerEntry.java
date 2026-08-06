/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Binary directory entry for a single GKV layer.
 * Defines the offsets and sizes for the Key and Value tensors.
 */
public record GKVLayerEntry(
    int layerId,
    long keyOffset,
    long valueOffset,
    long sizeBytes
) {
    public static final int ENTRY_SIZE = 32; // Aligned
}
