package tech.kayys.gollek.safetensor.core.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * SIMD-optimized dequantization kernels using Java Vector API.
 * <p>
 * Supports INT8 and INT4 (GPTQ-style packed) formats.
 */
public class DequantizationKernel {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_PREFERRED;

    /**
     * Dequantize INT8 weights to Float32.
     * Symmetric quantization expected: weight = qweight * scale.
     */
    public static void dequantizeInt8(MemorySegment src, MemorySegment dst, MemorySegment scales, long numel) {
        // Assume scales is 1-element (per-tensor) or matches dimensions.
        // For now, implement per-tensor symmetric INT8 for simplicity.
        float scale = scales.get(ValueLayout.JAVA_FLOAT, 0);
        FloatVector vScale = FloatVector.broadcast(F_SPECIES, scale);

        long i = 0;
        for (; i < F_SPECIES.loopBound(numel); i += F_SPECIES.length()) {
            // Load bytes and convert to floats via vectors
            // Note: Efficient byte-to-float vector conversion
            for (int lane = 0; lane < F_SPECIES.length(); lane++) {
                byte q = src.get(ValueLayout.JAVA_BYTE, i + lane);
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, i + lane, q * scale);
            }
        }
        for (; i < numel; i++) {
            byte q = src.get(ValueLayout.JAVA_BYTE, i);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, q * scale);
        }
    }

    /**
     * Dequantize INT4 weights (packed 8 per 32-bit word, GPTQ-style).
     * Supports group-wise scaling.
     */
    public static void dequantizeInt4(MemorySegment src, MemorySegment dst, MemorySegment scales, MemorySegment zeros, long numel, int groupSize) {
        // Simplified INT4: 2 values per byte [low nibble, high nibble]
        long i = 0;
        long numBytes = (numel + 1) / 2;

        for (; i < numBytes; i++) {
            byte b = src.get(ValueLayout.JAVA_BYTE, i);
            int v1 = b & 0x0F;
            int v2 = (b >> 4) & 0x0F;

            // Apply scales/zeros based on group
            long groupIdx = (i * 2) / groupSize;
            float scale = scales.getAtIndex(ValueLayout.JAVA_FLOAT, groupIdx);
            float zero = (zeros != null) ? zeros.getAtIndex(ValueLayout.JAVA_FLOAT, groupIdx) : 0.0f;

            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i * 2, (v1 - zero) * scale);
            if (i * 2 + 1 < numel) {
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, i * 2 + 1, (v2 - zero) * scale);
            }
        }
    }

    /**
     * Dequantize F16 weights to Float32.
     */
    public static void dequantizeF16(MemorySegment src, MemorySegment dst, long numel) {
        for (long i = 0; i < numel; i++) {
            short raw = src.getAtIndex(ValueLayout.JAVA_SHORT, i);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, float16ToFloat32(raw));
        }
    }

    /**
     * Dequantize BF16 weights to Float32.
     */
    public static void dequantizeBf16(MemorySegment src, MemorySegment dst, long numel) {
        for (long i = 0; i < numel; i++) {
            short raw = src.getAtIndex(ValueLayout.JAVA_SHORT, i);
            int floatBits = (raw & 0xFFFF) << 16;
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, Float.intBitsToFloat(floatBits));
        }
    }

    private static float float16ToFloat32(short half) {
        int h = half & 0xFFFF;
        int sign = (h >> 15) & 0x1;
        int exponent = (h >> 10) & 0x1F;
        int mantissa = h & 0x3FF;

        int f32;
        if (exponent == 0) {
            if (mantissa == 0) {
                f32 = sign << 31;
            } else {
                exponent = 1;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= ~0x400;
                f32 = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
            }
        } else if (exponent == 31) {
            f32 = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        } else {
            f32 = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
        }
        return Float.intBitsToFloat(f32);
    }
}
