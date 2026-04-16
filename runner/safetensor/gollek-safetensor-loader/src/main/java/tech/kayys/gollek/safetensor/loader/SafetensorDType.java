/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorDType.java
 * --------------------
 * Enumeration of every data type that the SafeTensors specification defines.
 * Each variant carries:
 *   - the canonical JSON string used in the header  (dtype field)
 *   - the byte-width of a single scalar element
 *   - whether the type is floating-point or integer
 *   - a human-readable display label
 *
 * The enum is used throughout the loader to:
 *   1. Parse the JSON header without string comparisons scattered across the code.
 *   2. Compute byte offsets and allocate correctly-sized off-heap segments.
 *   3. Drive dispatch in the tensor-view layer (e.g., asFloatBuffer vs asLongBuffer).
 *
 * References:
 *   https://github.com/huggingface/safetensors/blob/main/safetensors/src/lib.rs
 */
package tech.kayys.gollek.safetensor.loader;

import java.util.Locale;

/**
 * SafeTensors element data types.
 *
 * <p>
 * Maps directly to the {@code dtype} field in the SafeTensors JSON header.
 * Use {@link #fromJson(String)} to parse the string coming out of the file.
 */
public enum SafetensorDType {

    // ── Floating-point types ──────────────────────────────────────────────────

    /** Brain-Float 16 — 1 byte exponent, 7-bit mantissa. PyTorch default. */
    BF16("BF16", 2, true, "bfloat16"),

    /** IEEE 754 half-precision float. Common in GPU workloads. */
    F16("F16", 2, true, "float16"),

    /** IEEE 754 single-precision float. */
    F32("F32", 4, true, "float32"),

    /** IEEE 754 double-precision float. */
    F64("F64", 8, true, "float64"),

    // ── 8-bit float variants (micro-exponent formats) ────────────────────────

    /** OFP8 e4m3 — used in NVIDIA H100 / transformer engine. */
    F8_E4M3("F8_E4M3", 1, true, "float8_e4m3"),

    /** OFP8 e5m2 — used in NVIDIA H100 / transformer engine. */
    F8_E5M2("F8_E5M2", 1, true, "float8_e5m2"),

    // ── Integer types ─────────────────────────────────────────────────────────

    /** Signed 8-bit integer. */
    I8("I8", 1, false, "int8"),

    /** Signed 16-bit integer. */
    I16("I16", 2, false, "int16"),

    /** Signed 32-bit integer. */
    I32("I32", 4, false, "int32"),

    /** Signed 64-bit integer. */
    I64("I64", 8, false, "int64"),

    /** Unsigned 8-bit integer. */
    U8("U8", 1, false, "uint8"),

    /** Unsigned 16-bit integer. */
    U16("U16", 2, false, "uint16"),

    /** Unsigned 32-bit integer. */
    U32("U32", 4, false, "uint32"),

    /** Unsigned 64-bit integer. */
    U64("U64", 8, false, "uint64"),

    // ── Boolean ───────────────────────────────────────────────────────────────

    /** Boolean — stored as a single byte (0 or 1). */
    BOOL("BOOL", 1, false, "bool");

    // ─────────────────────────────────────────────────────────────────────────

    /** Exact string as it appears in the SafeTensors JSON header. */
    private final String jsonName;

    /** Byte width of one element. Used for offset arithmetic. */
    private final int byteSize;

    /**
     * {@code true} for floating-point dtypes, {@code false} for integers and bool.
     */
    private final boolean floatingPoint;

    /** Human-readable name (e.g., "float32"). */
    private final String displayName;

    // ─────────────────────────────────────────────────────────────────────────

    SafetensorDType(String jsonName, int byteSize, boolean floatingPoint, String displayName) {
        this.jsonName = jsonName;
        this.byteSize = byteSize;
        this.floatingPoint = floatingPoint;
        this.displayName = displayName;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /**
     * The canonical string used in the SafeTensors JSON header.
     * Example: {@code "BF16"}, {@code "F32"}, {@code "I8"}.
     */
    public String jsonName() {
        return jsonName;
    }

    /**
     * The byte-width of a single element of this dtype.
     * Use this to compute {@code byteOffset = elementOffset * dtype.byteSize()}.
     */
    public int byteSize() {
        return byteSize;
    }

    /**
     * {@code true} if this dtype represents a floating-point number
     * (BF16, F16, F32, F64, F8_*).
     */
    public boolean isFloatingPoint() {
        return floatingPoint;
    }

    /** Human-readable name for logging and metrics labels. */
    public String displayName() {
        return displayName;
    }

    // ── Factory ───────────────────────────────────────────────────────────────

    /**
     * Parse the {@code dtype} field from a SafeTensors JSON header.
     *
     * <p>
     * Matching is case-insensitive so that both {@code "F32"} and {@code "f32"}
     * are accepted, matching the lenient behaviour of the reference Python library.
     *
     * @param json the raw dtype string from the JSON header
     * @return the corresponding {@link SafetensorDType}
     * @throws IllegalArgumentException if the dtype is not recognised
     */
    public static SafetensorDType fromJson(String json) {
        if (json == null || json.isBlank()) {
            throw new IllegalArgumentException("dtype string is null or blank");
        }
        String normalised = json.strip().toUpperCase(Locale.ROOT);
        for (SafetensorDType dt : values()) {
            if (dt.jsonName.equals(normalised)) {
                return dt;
            }
        }
        throw new IllegalArgumentException(
                "Unknown SafeTensors dtype: '" + json + "'. "
                        + "Expected one of: BF16, F16, F32, F64, F8_E4M3, F8_E5M2, "
                        + "I8, I16, I32, I64, U8, U16, U32, U64, BOOL");
    }

    /**
     * Compute the total byte-count for a tensor described by this dtype and a
     * given shape. Overflow-safe: returns {@code -1} if the product exceeds
     * {@code Long.MAX_VALUE / byteSize}.
     *
     * @param shape dimension sizes
     * @return total byte count, or {@code -1} on overflow
     */
    public long totalBytes(long... shape) {
        long elements = 1L;
        for (long dim : shape) {
            if (dim < 0) {
                throw new IllegalArgumentException("Negative dimension: " + dim);
            }
            // Overflow check before multiplication
            if (elements != 0 && dim > Long.MAX_VALUE / elements) {
                return -1L;
            }
            elements *= dim;
        }
        long bytes = elements * byteSize;
        return bytes < 0 ? -1L : bytes;
    }

    @Override
    public String toString() {
        return jsonName + "(" + displayName + ", " + byteSize + "B)";
    }
}
