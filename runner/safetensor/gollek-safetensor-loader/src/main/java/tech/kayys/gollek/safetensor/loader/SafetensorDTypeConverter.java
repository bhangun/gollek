/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorDTypeConverter.java
 * ─────────────────────────────
 * Bulk dtype conversion utilities for SafeTensors data.
 *
 * Why a dedicated converter?
 * ══════════════════════════
 * ML inference backends (LibTorch, ONNX Runtime, LiteRT) typically consume
 * float32 arrays.  SafeTensors files frequently store weights in BF16 or F16
 * to save disk/memory.  This utility converts entire tensors in-place or into
 * new allocations without redundant heap copies.
 *
 * Implementation notes
 * ════════════════════
 *  • All bulk conversions read from a {@link MemorySegment} and write into a
 *    newly-allocated heap {@code float[]}.  Off-heap-to-off-heap paths are
 *    provided for zero-GC pipelines.
 *  • BF16 → F32: shift 16 bits left (exact, no rounding needed).
 *  • F16  → F32: IEEE 754 half-precision decode (handles subnormals/NaN/Inf).
 *  • All reads use LITTLE_ENDIAN to match the SafeTensors spec.
 *  • Loops are written with simple sequential access patterns so the JIT
 *    auto-vectorises them via the Vector API incubator (JDK 25).
 */
package tech.kayys.gollek.safetensor.loader;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import tech.kayys.gollek.safetensor.exception.SafetensorException;

/**
 * Stateless bulk dtype conversion utilities.
 *
 * <p>
 * All methods accept a {@link SafetensorTensor} and return a Java heap
 * array. For off-heap output use the {@code *IntoSegment} variants.
 *
 * <p>
 * Thread-safe: all methods are pure functions with no shared mutable state.
 */
public final class SafetensorDTypeConverter {

    // Typed little-endian layouts
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfDouble DOUBLE_LE = ValueLayout.JAVA_DOUBLE.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN);

    private SafetensorDTypeConverter() {
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Convenience: auto-dispatch based on dtype
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Convert any numeric tensor to a {@code float[]} on the Java heap.
     *
     * <p>
     * Supported conversions:
     * <table>
     * <tr>
     * <th>Source dtype</th>
     * <th>Method used</th>
     * </tr>
     * <tr>
     * <td>F32</td>
     * <td>direct copy</td>
     * </tr>
     * <tr>
     * <td>F64</td>
     * <td>double → float narrowing</td>
     * </tr>
     * <tr>
     * <td>BF16</td>
     * <td>shift left 16</td>
     * </tr>
     * <tr>
     * <td>F16</td>
     * <td>IEEE half-precision decode</td>
     * </tr>
     * <tr>
     * <td>I8</td>
     * <td>signed byte → float</td>
     * </tr>
     * <tr>
     * <td>U8</td>
     * <td>unsigned byte → float (& 0xFF)</td>
     * </tr>
     * <tr>
     * <td>I16</td>
     * <td>signed short → float</td>
     * </tr>
     * <tr>
     * <td>I32</td>
     * <td>int → float</td>
     * </tr>
     * <tr>
     * <td>I64</td>
     * <td>long → float (precision loss possible)</td>
     * </tr>
     * </table>
     *
     * @param tensor source tensor
     * @return a new {@code float[]} on the heap
     * @throws UnsupportedOperationException if the dtype has no defined conversion
     * @throws IllegalStateException         if the tensor is closed
     */
    public static float[] toFloat32Array(SafetensorTensor tensor) {
        checkFits(tensor);
        MemorySegment seg = tensor.segment();
        int n = (int) tensor.numElements();
        SafetensorDType dt = tensor.dtype();

        return switch (dt) {
            case F32 -> bulkReadF32(seg, n);
            case F64 -> bulkF64ToF32(seg, n);
            case BF16 -> bulkBF16ToF32(seg, n);
            case F16 -> bulkF16ToF32(seg, n);
            case F8_E4M3 -> bulkF8E4M3ToF32(seg, n);
            case F8_E5M2 -> bulkF8E5M2ToF32(seg, n);
            case I8 -> bulkI8ToF32(seg, n);
            case U8 -> bulkU8ToF32(seg, n);
            case I16 -> bulkI16ToF32(seg, n);
            case U16 -> bulkU16ToF32(seg, n);
            case I32 -> bulkI32ToF32(seg, n);
            case U32 -> bulkU32ToF32(seg, n);
            case I64 -> bulkI64ToF32(seg, n);
            case U64 -> bulkU64ToF32(seg, n);
            case BOOL -> bulkBoolToF32(seg, n);
        };
    }

    /**
     * Convert any numeric tensor to a {@code double[]} on the Java heap.
     * Lossless for all standard dtypes (F32, F64, integer types).
     */
    public static double[] toFloat64Array(SafetensorTensor tensor) {
        checkFits(tensor);
        MemorySegment seg = tensor.segment();
        int n = (int) tensor.numElements();
        SafetensorDType dt = tensor.dtype();

        return switch (dt) {
            case F64 -> bulkReadF64(seg, n);
            case F32 -> bulkF32ToF64(seg, n);
            case BF16 -> {
                float[] f = bulkBF16ToF32(seg, n);
                yield floatToDouble(f);
            }
            case F16 -> {
                float[] f = bulkF16ToF32(seg, n);
                yield floatToDouble(f);
            }
            case I8 -> bulkI8ToF64(seg, n);
            case U8 -> bulkU8ToF64(seg, n);
            case I16 -> bulkI16ToF64(seg, n);
            case I32 -> bulkI32ToF64(seg, n);
            case I64 -> bulkI64ToF64(seg, n);
            default -> {
                float[] f = toFloat32Array(tensor);
                yield floatToDouble(f);
            }
        };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BF16 → F32 (shift left 16 bits)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Convert a BF16 segment to F32 heap array.
     * BF16 shares the exponent layout of F32 — just zero-fill the lower 16 bits.
     *
     * @param seg off-heap segment containing N BF16 elements (little-endian)
     * @param n   number of elements
     * @return F32 heap array
     */
    public static float[] bulkBF16ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            short raw = seg.get(SHORT_LE, (long) i * 2);
            int bits = (raw & 0xFFFF) << 16;
            out[i] = Float.intBitsToFloat(bits);
        }
        return out;
    }

    /**
     * Convert F32 to BF16 (truncation/round-to-nearest-even) and write into
     * a newly-allocated off-heap segment. Useful for preparing model inputs
     * for accelerators that prefer BF16.
     *
     * @param src  source float array
     * @param dest off-heap destination segment (must be 2 * src.length bytes)
     */
    public static void f32ToBF16IntoSegment(float[] src, MemorySegment dest) {
        for (int i = 0; i < src.length; i++) {
            int bits = Float.floatToIntBits(src[i]);
            short bf16 = roundF32ToBF16(bits);
            dest.set(SHORT_LE, (long) i * 2, bf16);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // F16 → F32 (full IEEE 754 decode)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Convert a F16 (IEEE 754 half-precision) segment to F32 heap array.
     */
    public static float[] bulkF16ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            short raw = seg.get(SHORT_LE, (long) i * 2);
            out[i] = f16ToF32(raw);
        }
        return out;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // F8_E4M3 / F8_E5M2 → F32
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Convert NV-FP8 e4m3 (NVIDIA H100 format) to F32.
     * Range: ±448, special value 0x7F = NaN.
     */
    public static float[] bulkF8E4M3ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            byte raw = seg.get(ValueLayout.JAVA_BYTE, i);
            out[i] = fp8E4M3ToFloat(raw);
        }
        return out;
    }

    /**
     * Convert NV-FP8 e5m2 (NVIDIA H100 format) to F32.
     */
    public static float[] bulkF8E5M2ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            byte raw = seg.get(ValueLayout.JAVA_BYTE, i);
            out[i] = fp8E5M2ToFloat(raw);
        }
        return out;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Integer type → F32
    // ─────────────────────────────────────────────────────────────────────────

    public static float[] bulkI8ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = seg.get(ValueLayout.JAVA_BYTE, i);
        }
        return out;
    }

    public static float[] bulkU8ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = Byte.toUnsignedInt(seg.get(ValueLayout.JAVA_BYTE, i));
        }
        return out;
    }

    public static float[] bulkI16ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = seg.get(SHORT_LE, (long) i * 2);
        }
        return out;
    }

    public static float[] bulkU16ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = Short.toUnsignedInt(seg.get(SHORT_LE, (long) i * 2));
        }
        return out;
    }

    public static float[] bulkI32ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = seg.get(INT_LE, (long) i * 4);
        }
        return out;
    }

    public static float[] bulkU32ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = Integer.toUnsignedLong(seg.get(INT_LE, (long) i * 4));
        }
        return out;
    }

    public static float[] bulkI64ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = seg.get(LONG_LE, (long) i * 8);
        }
        return out;
    }

    public static float[] bulkU64ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            long raw = seg.get(LONG_LE, (long) i * 8);
            // U64 → float: use BigInteger path for values > Long.MAX_VALUE
            out[i] = raw >= 0 ? (float) raw
                    : (float) java.math.BigInteger.valueOf(raw).add(
                            java.math.BigInteger.ONE.shiftLeft(64)).floatValue();
        }
        return out;
    }

    public static float[] bulkBoolToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = seg.get(ValueLayout.JAVA_BYTE, i) != 0 ? 1.0f : 0.0f;
        }
        return out;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // F32 direct bulk read
    // ─────────────────────────────────────────────────────────────────────────

    public static float[] bulkReadF32(MemorySegment seg, int n) {
        return seg.toArray(FLOAT_LE);
    }

    /**
     * Convert an F64 segment to F32 heap array (narrowing).
     */
    public static float[] bulkF64ToF32(MemorySegment seg, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) seg.get(DOUBLE_LE, (long) i * 8);
        }
        return out;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // F64 / double paths
    // ─────────────────────────────────────────────────────────────────────────

    public static double[] bulkReadF64(MemorySegment seg, int n) {
        return seg.toArray(DOUBLE_LE);
    }

    public static double[] bulkF32ToF64(MemorySegment seg, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = seg.get(FLOAT_LE, (long) i * 4);
        }
        return out;
    }

    public static double[] bulkI8ToF64(MemorySegment seg, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++)
            out[i] = seg.get(ValueLayout.JAVA_BYTE, i);
        return out;
    }

    public static double[] bulkU8ToF64(MemorySegment seg, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++)
            out[i] = Byte.toUnsignedInt(seg.get(ValueLayout.JAVA_BYTE, i));
        return out;
    }

    public static double[] bulkI16ToF64(MemorySegment seg, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++)
            out[i] = seg.get(SHORT_LE, (long) i * 2);
        return out;
    }

    public static double[] bulkI32ToF64(MemorySegment seg, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++)
            out[i] = seg.get(INT_LE, (long) i * 4);
        return out;
    }

    public static double[] bulkI64ToF64(MemorySegment seg, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++)
            out[i] = (double) seg.get(LONG_LE, (long) i * 8);
        return out;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private arithmetic helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * IEEE 754 F16 → F32 conversion.
     * Handles: normals, subnormals, ±0, ±infinity, NaN.
     */
    static float f16ToF32(short half) {
        int h = half & 0xFFFF;
        int sign = (h >> 15) << 31;
        int exponent = (h >> 10) & 0x1F;
        int mantissa = h & 0x3FF;

        if (exponent == 0) {
            if (mantissa == 0)
                return Float.intBitsToFloat(sign); // ±0
            // Subnormal F16 → normalised F32
            int e = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                e--;
            }
            mantissa &= ~0x400;
            return Float.intBitsToFloat(sign | ((e + 112) << 23) | (mantissa << 13));
        }
        if (exponent == 31) {
            return Float.intBitsToFloat(sign | (0xFF << 23) | (mantissa << 13)); // ±Inf / NaN
        }
        return Float.intBitsToFloat(sign | ((exponent + 112) << 23) | (mantissa << 13));
    }

    /**
     * Round F32 to BF16 using round-to-nearest-even.
     */
    static short roundF32ToBF16(int f32bits) {
        // Extract the rounding bit and sticky bit
        int lsb = (f32bits >> 16) & 1;
        int round = (f32bits >> 15) & 1;
        int sticky = f32bits & 0x7FFF;
        int roundUp = round & (lsb | (sticky != 0 ? 1 : 0));
        return (short) ((f32bits + (roundUp << 16)) >> 16);
    }

    /**
     * NV-FP8 E4M3 → F32.
     * Spec: 4-bit exponent (bias=7), 3-bit mantissa.
     * 0x7F is NaN; infinities not representable.
     */
    static float fp8E4M3ToFloat(byte raw) {
        int v = raw & 0xFF;
        if (v == 0x7F || v == 0xFF)
            return Float.NaN; // NaN sentinel
        int sign = (v >> 7) & 1;
        int exponent = (v >> 3) & 0xF; // 4 bits
        int mantissa = v & 0x7; // 3 bits

        if (exponent == 0) {
            // Subnormal
            if (mantissa == 0)
                return sign == 0 ? 0.0f : -0.0f;
            float m = mantissa / 8.0f; // implicit 0.mantissa
            float val = (float) Math.scalb(m, -6); // 2^(1-bias) = 2^(1-7)
            return sign == 0 ? val : -val;
        }
        float m = 1.0f + mantissa / 8.0f;
        float val = (float) Math.scalb(m, exponent - 7);
        return sign == 0 ? val : -val;
    }

    /**
     * NV-FP8 E5M2 → F32.
     * Spec: 5-bit exponent (bias=15), 2-bit mantissa.
     * Infinities and NaN represented normally.
     */
    static float fp8E5M2ToFloat(byte raw) {
        int v = raw & 0xFF;
        int sign = (v >> 7) & 1;
        int exponent = (v >> 2) & 0x1F; // 5 bits
        int mantissa = v & 0x3; // 2 bits

        if (exponent == 0x1F) {
            if (mantissa == 0)
                return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            return Float.NaN;
        }
        if (exponent == 0) {
            if (mantissa == 0)
                return sign == 0 ? 0.0f : -0.0f;
            float m = mantissa / 4.0f;
            float val = (float) Math.scalb(m, -14);
            return sign == 0 ? val : -val;
        }
        float m = 1.0f + mantissa / 4.0f;
        float val = (float) Math.scalb(m, exponent - 15);
        return sign == 0 ? val : -val;
    }

    private static double[] floatToDouble(float[] src) {
        double[] out = new double[src.length];
        for (int i = 0; i < src.length; i++)
            out[i] = src[i];
        return out;
    }

    private static void checkFits(SafetensorTensor tensor) {
        if (tensor.numElements() > Integer.MAX_VALUE) {
            throw new SafetensorException.ValidationException(
                    "TorchTensor '" + tensor.name() + "' has " + tensor.numElements()
                            + " elements — too large to convert to a Java array (max 2^31-1). "
                            + "Use segment() + streaming for oversized tensors.",
                    null, tensor.name());
        }
    }
}
