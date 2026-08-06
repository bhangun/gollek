/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Defines the quantization levels for the GKV memory fabric.
 * Used for adaptive bandwidth and storage optimization.
 */
public enum QuantizationTier {
    FP16(0),     // Half-precision (Hot/VRAM)
    BF16(1),     // BFloat16 (Hot/TPU)
    INT8(2),     // 8-bit Integer (Warm/RAM)
    Q4_0(3),     // 4-bit Quantization (Llama.cpp style, Cold/Disk)
    Q4_1(4),     // 4-bit with extra precision
    EXTREME(5);  // Highly compressed archival format

    private final int id;

    QuantizationTier(int id) {
        this.id = id;
    }

    public int id() {
        return id;
    }

    public static QuantizationTier fromId(int id) {
        for (QuantizationTier t : values()) {
            if (t.id == id) return t;
        }
        return FP16;
    }
}
