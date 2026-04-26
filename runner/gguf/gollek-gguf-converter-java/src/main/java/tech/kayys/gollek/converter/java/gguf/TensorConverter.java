package tech.kayys.gollek.converter.java.gguf;

import java.lang.foreign.*;
import java.nio.ByteOrder;

/**
 * Low-level tensor data converters using JDK 25 FFM {@link MemorySegment}.
 *
 * <p>
 * All methods operate on off-heap memory to avoid heap pressure when
 * processing large model weight tensors.
 */
public final class TensorConverter {

    private TensorConverter() {
    }

    // ── BF16 → F32 ────────────────────────────────────────────────────────

    /**
     * Convert a BF16 tensor (brain-float 16) to F32 in-place (returning new
     * segment).
     * BF16 is simply the upper 16 bits of an IEEE 754 float32.
     */
    public static byte[] bf16ToF32(byte[] src, long numElements) {
        byte[] dst = new byte[(int) (numElements * 4)];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment srcSeg = arena.allocate(src.length);
            MemorySegment dstSeg = arena.allocate(dst.length);
            MemorySegment.copy(src, 0, srcSeg, ValueLayout.JAVA_BYTE, 0, src.length);

            for (long i = 0; i < numElements; i++) {
                short bf16 = srcSeg.get(
                        ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        i * 2);
                // Shift left by 16 to reinterpret as F32
                float f32 = Float.intBitsToFloat((bf16 & 0xFFFF) << 16);
                dstSeg.set(
                        ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        i * 4, f32);
            }
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_BYTE, 0, dst, 0, dst.length);
        }
        return dst;
    }

    // ── F16 → F32 ─────────────────────────────────────────────────────────

    /**
     * Convert IEEE 754 half-precision (F16) to F32.
     */
    public static byte[] f16ToF32(byte[] src, long numElements) {
        byte[] dst = new byte[(int) (numElements * 4)];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment srcSeg = arena.allocate(src.length);
            MemorySegment dstSeg = arena.allocate(dst.length);
            MemorySegment.copy(src, 0, srcSeg, ValueLayout.JAVA_BYTE, 0, src.length);

            for (long i = 0; i < numElements; i++) {
                short h = srcSeg.get(
                        ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        i * 2);
                float f = halfToFloat(h);
                dstSeg.set(
                        ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        i * 4, f);
            }
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_BYTE, 0, dst, 0, dst.length);
        }
        return dst;
    }

    // ── F32 → BF16 ────────────────────────────────────────────────────────

    /**
     * Convert F32 to BF16 by truncating the lower 16 mantissa bits.
     * Uses round-to-nearest-even for best accuracy.
     */
    public static byte[] f32ToBf16(byte[] src, long numElements) {
        byte[] dst = new byte[(int) (numElements * 2)];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment srcSeg = arena.allocate(src.length);
            MemorySegment dstSeg = arena.allocate(dst.length);
            MemorySegment.copy(src, 0, srcSeg, ValueLayout.JAVA_BYTE, 0, src.length);

            for (long i = 0; i < numElements; i++) {
                float f = srcSeg.get(
                        ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), i * 4);
                int bits = Float.floatToIntBits(f);
                // Round-to-nearest-even: add 0x7FFF + round bit
                int roundBit = (bits >> 16) & 1;
                int rounded = bits + 0x7FFF + roundBit;
                short bf16 = (short) (rounded >>> 16);
                dstSeg.set(
                        ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        i * 2, bf16);
            }
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_BYTE, 0, dst, 0, dst.length);
        }
        return dst;
    }

    // ── F32 → F16 ─────────────────────────────────────────────────────────

    public static byte[] f32ToF16(byte[] src, long numElements) {
        byte[] dst = new byte[(int) (numElements * 2)];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment srcSeg = arena.allocate(src.length);
            MemorySegment dstSeg = arena.allocate(dst.length);
            MemorySegment.copy(src, 0, srcSeg, ValueLayout.JAVA_BYTE, 0, src.length);

            for (long i = 0; i < numElements; i++) {
                float f = srcSeg.get(
                        ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        i * 4);
                short h = floatToHalf(f);
                dstSeg.set(
                        ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        i * 2, h);
            }
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_BYTE, 0, dst, 0, dst.length);
        }
        return dst;
    }

    // ── Q8_0 quantization (scalar reference) ──────────────────────────────

    /**
     * Quantize F32 data to Q8_0 format.
     * Block layout: [2-byte F16 scale][32 × int8 weights]
     * 
     * <p>FIXED: Uses C-compatible rounding (round-half-away-from-zero) instead of
     * Java's Math.round (round-half-to-even) to match llama.cpp reference.</p>
     */
    public static byte[] quantizeQ8_0(byte[] srcF32, long numElements) {
        if (numElements % 32 != 0)
            throw new IllegalArgumentException("Q8_0 requires element count % 32 == 0");
        long numBlocks = numElements / 32;
        // Each block = 2 (scale F16) + 32 (int8) = 34 bytes
        byte[] dst = new byte[(int) (numBlocks * 34)];

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment srcSeg = arena.allocate(srcF32.length);
            MemorySegment dstSeg = arena.allocate(dst.length);
            MemorySegment.copy(srcF32, 0, srcSeg, ValueLayout.JAVA_BYTE, 0, srcF32.length);

            for (long b = 0; b < numBlocks; b++) {
                // Find max absolute value in the block
                float maxAbs = 0f;
                for (int i = 0; i < 32; i++) {
                    float v = srcSeg.get(
                            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                            (b * 32 + i) * 4);
                    maxAbs = Math.max(maxAbs, Math.abs(v));
                }
                float scale = maxAbs / 127f;
                float invScale = (scale == 0f) ? 0f : 1f / scale;

                // Write F16 scale
                short scaleF16 = floatToHalf(scale);
                dstSeg.set(
                        ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        b * 34, scaleF16);

                // Write quantized weights with C-compatible rounding
                for (int i = 0; i < 32; i++) {
                    float v = srcSeg.get(
                            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                            (b * 32 + i) * 4);
                    // FIXED: Use nearestInt() matching C's (int)(x + 0.5 * sign(x))
                    int qi = nearestInt(v * invScale);
                    qi = Math.clamp(qi, -127, 127);
                    dstSeg.set(ValueLayout.JAVA_BYTE, b * 34 + 2 + i, (byte) qi);
                }
            }
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_BYTE, 0, dst, 0, dst.length);
        }
        return dst;
    }

    // ── Q4_0 quantization ─────────────────────────────────────────────────

    /**
     * Quantize F32 to Q4_0 format.
     * Block layout: [2-byte F16 scale][16 bytes of packed 4-bit weights]
     * 
     * <p>FIXED: Q4_0 nibble overflow - clamp to 0-15, not 0-16.
     * A 4-bit value can only hold 0-15; clamping to 16 would overflow into next nibble.</p>
     */
    public static byte[] quantizeQ4_0(byte[] srcF32, long numElements) {
        if (numElements % 32 != 0)
            throw new IllegalArgumentException("Q4_0 requires element count % 32 == 0");
        long numBlocks = numElements / 32;
        byte[] dst = new byte[(int) (numBlocks * 18)]; // 2 + 16

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment srcSeg = arena.allocate(srcF32.length);
            MemorySegment dstSeg = arena.allocate(dst.length);
            MemorySegment.copy(srcF32, 0, srcSeg, ValueLayout.JAVA_BYTE, 0, srcF32.length);

            for (long b = 0; b < numBlocks; b++) {
                float maxAbs = 0f;
                for (int i = 0; i < 32; i++) {
                    float v = srcSeg.get(
                            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                            (b * 32 + i) * 4);
                    maxAbs = Math.max(maxAbs, Math.abs(v));
                }
                float scale = maxAbs / 7f;
                float invScale = (scale == 0f) ? 0f : 1f / scale;

                // Write F16 scale
                dstSeg.set(
                        ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        b * 18, floatToHalf(scale));

                // Pack two 4-bit values per byte
                for (int i = 0; i < 16; i++) {
                    float v0 = srcSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED
                            .withOrder(ByteOrder.LITTLE_ENDIAN), (b * 32 + i * 2) * 4);
                    float v1 = srcSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED
                            .withOrder(ByteOrder.LITTLE_ENDIAN), (b * 32 + i * 2 + 1) * 4);
                    // FIXED: clamp to 0-15, not 0-16 (nibble overflow bug)
                    int q0 = Math.clamp(nearestInt(v0 * invScale) + 8, 0, 15);
                    int q1 = Math.clamp(nearestInt(v1 * invScale) + 8, 0, 15);
                    dstSeg.set(ValueLayout.JAVA_BYTE, b * 18 + 2 + i,
                            (byte) ((q0) | (q1 << 4)));
                }
            }
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_BYTE, 0, dst, 0, dst.length);
        }
        return dst;
    }

    // ── K-Quant implementations ───────────────────────────────────────────

    /**
     * Quantize F32 to Q2_K format (2-bit K-quant, 256 elements per super-block).
     * Block layout: 84 bytes per 256 elements.
     */
    public static byte[] quantizeQ2_K(byte[] srcF32, long numElements) {
        if (numElements % 256 != 0)
            throw new IllegalArgumentException("Q2_K requires element count % 256 == 0");
        long numBlocks = numElements / 256;
        byte[] dst = new byte[(int) (numBlocks * 84)];

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment srcSeg = arena.allocate(srcF32.length);
            MemorySegment dstSeg = arena.allocate(dst.length);
            MemorySegment.copy(srcF32, 0, srcSeg, ValueLayout.JAVA_BYTE, 0, srcF32.length);

            for (long b = 0; b < numBlocks; b++) {
                float[] scales = new float[16];
                float[] mins = new float[16];

                for (int sb = 0; sb < 16; sb++) {
                    float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
                    for (int i = 0; i < 16; i++) {
                        float v = srcSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED
                                .withOrder(ByteOrder.LITTLE_ENDIAN), (b * 256 + sb * 16 + i) * 4);
                        min = Math.min(min, v);
                        max = Math.max(max, v);
                    }
                    scales[sb] = (max - min) / 3.0f;
                    mins[sb] = min;
                }

                for (int sb = 0; sb < 16; sb += 2) {
                    dstSeg.set(ValueLayout.JAVA_BYTE, b * 84 + sb / 2,
                            (byte) ((encodeScaleQ2K(scales[sb]) << 4) | encodeScaleQ2K(scales[sb + 1])));
                    dstSeg.set(ValueLayout.JAVA_BYTE, b * 84 + 8 + sb / 2,
                            (byte) ((encodeMinQ2K(mins[sb]) << 4) | encodeMinQ2K(mins[sb + 1])));
                }

                for (int sb = 0; sb < 16; sb++) {
                    float scale = scales[sb] == 0f ? 1e-5f : scales[sb];
                    float invScale = 1f / scale;
                    for (int i = 0; i < 16; i += 4) {
                        int packed = 0;
                        for (int j = 0; j < 4; j++) {
                            float v = srcSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED
                                    .withOrder(ByteOrder.LITTLE_ENDIAN), (b * 256 + sb * 16 + i + j) * 4);
                            int q = Math.clamp(nearestInt((v - mins[sb]) * invScale), 0, 3);
                            packed |= (q << (j * 2));
                        }
                        dstSeg.set(ValueLayout.JAVA_BYTE, b * 84 + 16 + sb * 2 + i / 4, (byte) packed);
                    }
                }
            }
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_BYTE, 0, dst, 0, dst.length);
        }
        return dst;
    }

    /**
     * Quantize F32 to Q4_K format (4-bit K-quant, 256 elements per super-block).
     * Block layout: 144 bytes per 256 elements.
     */
    public static byte[] quantizeQ4_K(byte[] srcF32, long numElements) {
        if (numElements % 256 != 0)
            throw new IllegalArgumentException("Q4_K requires element count % 256 == 0");
        long numBlocks = numElements / 256;
        byte[] dst = new byte[(int) (numBlocks * 144)];

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment srcSeg = arena.allocate(srcF32.length);
            MemorySegment dstSeg = arena.allocate(dst.length);
            MemorySegment.copy(srcF32, 0, srcSeg, ValueLayout.JAVA_BYTE, 0, srcF32.length);

            for (long b = 0; b < numBlocks; b++) {
                float[] scales = new float[8];
                for (int sb = 0; sb < 8; sb++) {
                    float maxAbs = 0f;
                    for (int i = 0; i < 32; i++) {
                        float v = srcSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED
                                .withOrder(ByteOrder.LITTLE_ENDIAN), (b * 256 + sb * 32 + i) * 4);
                        maxAbs = Math.max(maxAbs, Math.abs(v));
                    }
                    scales[sb] = maxAbs / 7f;
                }

                packScales6bit(dstSeg, b * 144, scales);

                for (int sb = 0; sb < 8; sb++) {
                    float scale = scales[sb] == 0f ? 1e-5f : scales[sb];
                    float invScale = 1f / scale;
                    for (int i = 0; i < 32; i += 2) {
                        float v0 = srcSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED
                                .withOrder(ByteOrder.LITTLE_ENDIAN), (b * 256 + sb * 32 + i) * 4);
                        float v1 = srcSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED
                                .withOrder(ByteOrder.LITTLE_ENDIAN), (b * 256 + sb * 32 + i + 1) * 4);
                        int q0 = Math.clamp(nearestInt(v0 * invScale) + 8, 0, 15);
                        int q1 = Math.clamp(nearestInt(v1 * invScale) + 8, 0, 15);
                        dstSeg.set(ValueLayout.JAVA_BYTE, b * 144 + 6 + sb * 16 + i / 2,
                                (byte) ((q0) | (q1 << 4)));
                    }
                }
            }
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_BYTE, 0, dst, 0, dst.length);
        }
        return dst;
    }

    /**
     * Quantize F32 to Q5_K format (5-bit K-quant, 256 elements per super-block).
     * Block layout: 176 bytes per 256 elements.
     */
    public static byte[] quantizeQ5_K(byte[] srcF32, long numElements) {
        if (numElements % 256 != 0)
            throw new IllegalArgumentException("Q5_K requires element count % 256 == 0");
        long numBlocks = numElements / 256;
        byte[] dst = new byte[(int) (numBlocks * 176)];

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment srcSeg = arena.allocate(srcF32.length);
            MemorySegment dstSeg = arena.allocate(dst.length);
            MemorySegment.copy(srcF32, 0, srcSeg, ValueLayout.JAVA_BYTE, 0, srcF32.length);

            for (long b = 0; b < numBlocks; b++) {
                float[] scales = new float[8];
                for (int sb = 0; sb < 8; sb++) {
                    float maxAbs = 0f;
                    for (int i = 0; i < 32; i++) {
                        float v = srcSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED
                                .withOrder(ByteOrder.LITTLE_ENDIAN), (b * 256 + sb * 32 + i) * 4);
                        maxAbs = Math.max(maxAbs, Math.abs(v));
                    }
                    scales[sb] = maxAbs / 15f;
                }

                packScales6bit(dstSeg, b * 176, scales);

                for (int sb = 0; sb < 8; sb++) {
                    float scale = scales[sb] == 0f ? 1e-5f : scales[sb];
                    float invScale = 1f / scale;
                    for (int i = 0; i < 32; i++) {
                        float v = srcSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED
                                .withOrder(ByteOrder.LITTLE_ENDIAN), (b * 256 + sb * 32 + i) * 4);
                        int q = Math.clamp(nearestInt(v * invScale) + 16, 0, 31);
                        dstSeg.set(ValueLayout.JAVA_BYTE, b * 176 + 6 + sb * 32 + i, (byte) (q & 0x0F));
                        int hbIdx = sb * 32 + i;
                        byte hb = (byte) ((q >> 4) & 1);
                        byte ex = dstSeg.get(ValueLayout.JAVA_BYTE, b * 176 + 134 + hbIdx / 8);
                        dstSeg.set(ValueLayout.JAVA_BYTE, b * 176 + 134 + hbIdx / 8, (byte) (ex | (hb << (hbIdx % 8))));
                    }
                }
            }
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_BYTE, 0, dst, 0, dst.length);
        }
        return dst;
    }

    /**
     * Quantize F32 to Q6_K format (6-bit K-quant, 256 elements per super-block).
     * Block layout: 210 bytes per 256 elements.
     */
    public static byte[] quantizeQ6_K(byte[] srcF32, long numElements) {
        if (numElements % 256 != 0)
            throw new IllegalArgumentException("Q6_K requires element count % 256 == 0");
        long numBlocks = numElements / 256;
        byte[] dst = new byte[(int) (numBlocks * 210)];

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment srcSeg = arena.allocate(srcF32.length);
            MemorySegment dstSeg = arena.allocate(dst.length);
            MemorySegment.copy(srcF32, 0, srcSeg, ValueLayout.JAVA_BYTE, 0, srcF32.length);

            for (long b = 0; b < numBlocks; b++) {
                float[] scales = new float[16];
                for (int sb = 0; sb < 16; sb++) {
                    float maxAbs = 0f;
                    for (int i = 0; i < 16; i++) {
                        float v = srcSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED
                                .withOrder(ByteOrder.LITTLE_ENDIAN), (b * 256 + sb * 16 + i) * 4);
                        maxAbs = Math.max(maxAbs, Math.abs(v));
                    }
                    scales[sb] = maxAbs / 31f;
                }

                for (int sb = 0; sb < 16; sb += 2) {
                    dstSeg.set(ValueLayout.JAVA_BYTE, b * 210 + sb / 2,
                            (byte) ((encodeScaleQ6K(scales[sb]) << 4) | encodeScaleQ6K(scales[sb + 1])));
                }

                for (int sb = 0; sb < 16; sb++) {
                    float scale = scales[sb] == 0f ? 1e-5f : scales[sb];
                    float invScale = 1f / scale;
                    for (int i = 0; i < 16; i++) {
                        float v = srcSeg.get(ValueLayout.JAVA_FLOAT_UNALIGNED
                                .withOrder(ByteOrder.LITTLE_ENDIAN), (b * 256 + sb * 16 + i) * 4);
                        int q = Math.clamp(nearestInt(v * invScale) + 32, 0, 63);
                        dstSeg.set(ValueLayout.JAVA_BYTE, b * 210 + 8 + sb * 16 + i, (byte) (q & 0x0F));
                        int hbIdx = sb * 16 + i;
                        byte hb = (byte) ((q >> 4) & 0x03);
                        byte ex = dstSeg.get(ValueLayout.JAVA_BYTE, b * 210 + 200 + hbIdx / 4);
                        dstSeg.set(ValueLayout.JAVA_BYTE, b * 210 + 200 + hbIdx / 4, (byte) (ex | (hb << ((hbIdx % 4) * 2))));
                    }
                }
            }
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_BYTE, 0, dst, 0, dst.length);
        }
        return dst;
    }

    // ── Conversion helpers ────────────────────────────────────────────────

    /**
     * Round to nearest int matching C's (int)(x + 0.5 * sign(x)) behavior.
     * This is NOT Math.round (which uses round-half-to-even).
     */
    private static int nearestInt(float x) {
        return x >= 0 ? (int) (x + 0.5f) : (int) (x - 0.5f);
    }

    private static void packScales6bit(MemorySegment dstSeg, long offset, float[] scales) {
        long packed = 0;
        for (int i = 0; i < 8; i++) {
            int q = Math.clamp((int) (scales[i] * 63.99f / 7f), 0, 63);
            packed |= ((long) q) << (i * 6);
        }
        dstSeg.set(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), offset, packed);
    }

    private static int encodeScaleQ2K(float scale) {
        return Math.clamp(nearestInt(scale * 15f / 7f), 0, 15);
    }

    private static int encodeMinQ2K(float min) {
        return Math.clamp(nearestInt(min * 15f / 7f), 0, 15);
    }

    private static int encodeScaleQ6K(float scale) {
        return Math.clamp(nearestInt(scale * 15f / 7f), 0, 15);
    }

    /** IEEE 754 half-precision float → Java float (pure Java). */
    public static float halfToFloat(short h) {
        int bits = h & 0xFFFF;
        int sign = (bits >>> 15) & 1;
        int exp = (bits >>> 10) & 0x1F;
        int mant = bits & 0x3FF;
        if (exp == 0) {
            // subnormal
            if (mant == 0)
                return sign == 0 ? 0f : -0f;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x400;
        } else if (exp == 31) {
            return mant == 0 ? (sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY)
                    : Float.NaN;
        }
        int f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        return Float.intBitsToFloat(f32);
    }

    /** Java float → IEEE 754 half-precision (nearest-even rounding). */
    public static short floatToHalf(float f) {
        int bits = Float.floatToIntBits(f);
        int sign = (bits >>> 31) & 1;
        int exp = ((bits >>> 23) & 0xFF) - 127 + 15;
        int mant = (bits >>> 13) & 0x3FF;
        if (exp <= 0) {
            if (exp < -10)
                return (short) (sign << 15);
            mant = ((bits & 0x7FFFFF) | 0x800000) >> (1 - exp);
            if ((mant & 0x1000) != 0)
                mant += 0x2000;
            return (short) ((sign << 15) | (mant >> 13));
        } else if (exp == 0xFF - (127 - 15)) {
            if (mant == 0)
                return (short) ((sign << 15) | 0x7C00);
            return (short) ((sign << 15) | 0x7C00 | (mant >> 1) | 1);
        } else if (exp > 30) {
            return (short) ((sign << 15) | 0x7C00); // infinity
        }
        return (short) ((sign << 15) | (exp << 10) | mant);
    }

    /**
     * Decide a sensible target type for a HF tensor given a global quant target.
     * 
     * <p>FIXED: Bias tensors must stay F32 to avoid quantization artifacts.</p>
     * Embeddings and norms are kept at higher precision for quality.
     */
    public static GgmlType targetType(String tensorName, GgmlType globalQuant) {
        // FIXED: Bias tensors must stay F32 to avoid quantization artifacts
        if (tensorName.endsWith(".bias")) {
            return GgmlType.F32;
        }

        // Keep embeddings and output projection at higher precision
        if (tensorName.contains("embed_tokens") ||
                tensorName.contains("lm_head") ||
                tensorName.contains("norm") ||
                tensorName.contains("layernorm")) {
            return switch (globalQuant) {
                case Q4_0, Q4_1, Q5_0, Q5_1,
                        Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K ->
                    GgmlType.F32;
                default -> globalQuant;
            };
        }
        return globalQuant;
    }
}
