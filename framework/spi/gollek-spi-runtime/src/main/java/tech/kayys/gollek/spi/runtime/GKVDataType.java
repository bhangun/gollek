/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Binary data types supported by the GKV (Gollek KV) format.
 * Defines the precision and quantization level for token memory.
 */
public enum GKVDataType {
    FP32(0, 4),   // 32-bit float
    FP16(1, 2),   // 16-bit float (standard)
    BF16(2, 2),   // Brain float 16
    INT8(3, 1),   // 8-bit integer
    Q4_0(4, 0),   // 4-bit quantized (GGML style)
    Q4_1(5, 0),   // 4-bit quantized (Type 1)
    Q5_0(6, 0),   // 5-bit quantized
    Q8_0(7, 1);   // 8-bit quantized

    private final int id;
    private final int sizeBytes;

    GKVDataType(int id, int sizeBytes) {
        this.id = id;
        this.sizeBytes = sizeBytes;
    }

    public int id() { return id; }
    public int sizeBytes() { return sizeBytes; }

    public static GKVDataType fromId(int id) {
        for (GKVDataType type : values()) {
            if (type.id == id) return type;
        }
        throw new IllegalArgumentException("Unknown GKV DataType ID: " + id);
    }
}
