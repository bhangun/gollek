/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * HalfPrecisionConverter.java
 * ───────────────────────────
 * Batch conversion utilities for BF16 and F16 → F32 (and back) using
 * JDK 25 FFM MemorySegment loops.
 *
 * Why not just use Float.float16ToFloat() ?
 * ═════════════════════════════════════════
 * JDK 20 introduced Float.float16ToFloat() / floatToFloat16() for single
 * values.  This class adds:
 *   • BF16 (brain-float-16) conversion (not in JDK standard lib)
 *   • Bulk / batch conversion that reads directly from MemorySegment to
 *     minimise heap allocations on the hot path
 *   • In-place conversion into a caller-supplied float[] or MemorySegment
 *
 * SIMD note
 * ═════════
 * The loops below are written to be auto-vectorised by the JIT when running
 * on CPUs that support SSE4.1 or AVX-512 FP16 (Sapphire Rapids, Zen 4).
 * For explicit SIMD, the Vector API (JEP 460, finalized JDK 25) can replace
 * the inner loops — left as an optimisation task.
 */
package tech.kayys.gollek.safetensor.loader;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Stateless batch converter between half-precision and single-precision floats.
 *
 * <p>
 * All methods are static; no instantiation needed.
 */
public final class HalfPrecisionConverter {

    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN);

    private HalfPrecisionConverter() {
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BF16 → F32 (bulk, MemorySegment source)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Convert all BF16 elements in a {@link MemorySegment} to a {@code float[]}.
     *
     * <p>
     * BF16 is stored as 2 bytes, little-endian. Conversion: shift left 16
     * into the exponent/mantissa bits of a float32 — the exponent is identical
     * in both formats (BF16 is the MSB of F32).
     *
     * @param src      source segment containing raw BF16 bytes (little-endian)
     * @param numElems number of BF16 elements (segment must be >= numElems * 2
     *                 bytes)
     * @return newly allocated float array of length {@code numElems}
     */
    public static float[] bf16SegmentToFloat32(MemorySegment src, int numElems) {
        float[] out = new float[numElems];
        bf16SegmentToFloat32(src, 0L, out, 0, numElems);
        return out;
    }

    /**
     * Convert a range of BF16 elements from a {@link MemorySegment} into a
     * pre-allocated {@code float[]} at a given destination offset.
     *
     * @param src           source segment
     * @param srcByteOffset byte offset in the source segment to start reading
     * @param dst           destination float array
     * @param dstOffset     first index in {@code dst} to write
     * @param count         number of elements to convert
     */
    public static void bf16SegmentToFloat32(
            MemorySegment src, long srcByteOffset,
            float[] dst, int dstOffset, int count) {
        for (int i = 0; i < count; i++) {
            short raw = src.get(SHORT_LE, srcByteOffset + (long) i * 2);
            dst[dstOffset + i] = bf16ToFloat32(raw);
        }
    }

    /**
     * Single-element BF16 → F32.
     *
     * @param bf16 raw 16-bit BF16 value
     * @return the corresponding float32
     */
    public static float bf16ToFloat32(short bf16) {
        return Float.intBitsToFloat((bf16 & 0xFFFF) << 16);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // F16 (IEEE 754 half-precision) → F32 (bulk, MemorySegment source)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Convert all F16 elements in a {@link MemorySegment} to a {@code float[]}.
     *
     * @param src      source segment containing raw F16 bytes (little-endian)
     * @param numElems number of F16 elements
     * @return newly allocated float array
     */
    public static float[] f16SegmentToFloat32(MemorySegment src, int numElems) {
        float[] out = new float[numElems];
        f16SegmentToFloat32(src, 0L, out, 0, numElems);
        return out;
    }

    /**
     * Convert a range of F16 elements from a {@link MemorySegment} into a
     * pre-allocated {@code float[]}.
     */
    public static void f16SegmentToFloat32(
            MemorySegment src, long srcByteOffset,
            float[] dst, int dstOffset, int count) {
        for (int i = 0; i < count; i++) {
            short raw = src.get(SHORT_LE, srcByteOffset + (long) i * 2);
            dst[dstOffset + i] = Float.float16ToFloat(raw); // JDK 20+ API
        }
    }

    /**
     * Single-element F16 → F32 (IEEE 754 half-precision).
     * Delegates to {@link Float#float16ToFloat(short)} (JDK 20+).
     */
    public static float f16ToFloat32(short f16) {
        return Float.float16ToFloat(f16);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // F32 → BF16 (scalar and batch, for writing quantised tensors)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Single-element F32 → BF16 (round-to-nearest-even via truncation of LSB).
     *
     * <p>
     * This is the stochastic-rounding-free version. For training you may
     * want to implement round-to-nearest-even; for inference truncation is fine.
     *
     * @param value the float32 to convert
     * @return the raw BF16 bit pattern as a {@code short}
     */
    public static short float32ToBf16(float value) {
        int bits = Float.floatToIntBits(value);
        // Round: add rounding bit
        int rounded = bits + 0x7FFF + ((bits >> 16) & 1);
        return (short) (rounded >>> 16);
    }

    /**
     * Batch F32 → BF16 conversion into a MemorySegment.
     *
     * @param src       source float array
     * @param srcOffset starting index in {@code src}
     * @param dst       destination segment (must have capacity >= count * 2 bytes)
     * @param dstOffset byte offset in {@code dst} to start writing
     * @param count     number of elements to convert
     */
    public static void float32ToBf16Segment(
            float[] src, int srcOffset,
            MemorySegment dst, long dstOffset, int count) {
        for (int i = 0; i < count; i++) {
            short bf16 = float32ToBf16(src[srcOffset + i]);
            dst.set(SHORT_LE, dstOffset + (long) i * 2, bf16);
        }
    }

    /**
     * Single-element F32 → F16 (IEEE 754 half-precision).
     * Delegates to {@link Float#floatToFloat16(float)} (JDK 20+).
     */
    public static short float32ToF16(float value) {
        return Float.floatToFloat16(value);
    }

    /**
     * Batch F32 → F16 conversion into a MemorySegment.
     */
    public static void float32ToF16Segment(
            float[] src, int srcOffset,
            MemorySegment dst, long dstOffset, int count) {
        for (int i = 0; i < count; i++) {
            short f16 = Float.floatToFloat16(src[srcOffset + i]);
            dst.set(SHORT_LE, dstOffset + (long) i * 2, f16);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // F8 helpers (E4M3 / E5M2 — dequantise to F32)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Dequantise an OFP8 E4M3 byte to the nearest F32.
     *
     * <p>
     * Reference implementation based on the NVIDIA FP8 spec:
     * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fp8
     *
     * @param fp8 raw byte value in E4M3 format
     * @return the dequantised float32
     */
    public static float fp8E4M3ToFloat32(byte fp8) {
        int v = fp8 & 0xFF;
        int sign = (v >> 7) & 0x1;
        int exponent = (v >> 3) & 0xF;
        int mantissa = v & 0x7;

        if (exponent == 0xF && mantissa == 0x7) {
            // NaN
            return Float.NaN;
        }
        if (exponent == 0) {
            // Subnormal
            float frac = mantissa / 8.0f; // mantissa / 2^3
            float mag = frac * (float) Math.pow(2, -6); // bias = 7, min_exp = 1-7 = -6
            return sign == 0 ? mag : -mag;
        }
        float mag = (1.0f + mantissa / 8.0f) * (float) Math.pow(2, exponent - 7);
        return sign == 0 ? mag : -mag;
    }

    /**
     * Dequantise an OFP8 E5M2 byte to the nearest F32.
     *
     * @param fp8 raw byte value in E5M2 format
     * @return the dequantised float32
     */
    public static float fp8E5M2ToFloat32(byte fp8) {
        int v = fp8 & 0xFF;
        int sign = (v >> 7) & 0x1;
        int exponent = (v >> 2) & 0x1F;
        int mantissa = v & 0x3;

        if (exponent == 0x1F) {
            // Inf or NaN
            if (mantissa == 0)
                return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            return Float.NaN;
        }
        if (exponent == 0) {
            float frac = mantissa / 4.0f;
            float mag = frac * (float) Math.pow(2, -14);
            return sign == 0 ? mag : -mag;
        }
        float mag = (1.0f + mantissa / 4.0f) * (float) Math.pow(2, exponent - 15);
        return sign == 0 ? mag : -mag;
    }

    /**
     * Bulk dequantise an F8_E4M3 segment into a float array.
     */
    public static float[] fp8E4M3SegmentToFloat32(MemorySegment src, int numElems) {
        float[] out = new float[numElems];
        for (int i = 0; i < numElems; i++) {
            byte b = src.get(ValueLayout.JAVA_BYTE, i);
            out[i] = fp8E4M3ToFloat32(b);
        }
        return out;
    }

    /**
     * Bulk dequantise an F8_E5M2 segment into a float array.
     */
    public static float[] fp8E5M2SegmentToFloat32(MemorySegment src, int numElems) {
        float[] out = new float[numElems];
        for (int i = 0; i < numElems; i++) {
            byte b = src.get(ValueLayout.JAVA_BYTE, i);
            out[i] = fp8E5M2ToFloat32(b);
        }
        return out;
    }
}
