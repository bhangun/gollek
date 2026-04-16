package tech.kayys.gollek.qlora.binding;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.jboss.logging.Logger;

/**
 * Pure-Java CPU fallback for QLoRA fused matmul.
 *
 * <p>Simulates the NF4 base + INT4 LoRA fused GEMM in fp32 arithmetic.
 * NF4 packing is approximated by treating the buffer as fp32 (unpacked).
 * Correct output, not performance-optimised. Used for CI and development.
 */
public final class QLoraCpuFallback {

    private static final Logger LOG = Logger.getLogger(QLoraCpuFallback.class);
    private static final float[] NF4_LOOKUP = {
            -1.0f, -0.69619286f, -0.52507305f, -0.3949175f,
            -0.28444138f, -0.18477343f, -0.091050036f, 0.0f,
            0.0795803f, 0.1609302f, 0.2461123f, 0.33791524f,
            0.44070983f, 0.562617f, 0.72295684f, 1.0f
    };

    private QLoraCpuFallback() {}

    /**
     * Fused matmul: out = (W_base + scale * B × A) × input  (fp32 approximation)
     */
    public static int fusedMatmul(
            MemorySegment out,
            MemorySegment wNf4,
            MemorySegment loraB,
            MemorySegment loraA,
            MemorySegment input,
            float loraScale,
            int M, int N, int K, int R) {

        LOG.debug("QLoRA CPU fallback (native unavailable)");

        boolean packedNf4 = wNf4.byteSize() < (long) N * K * 4L;
        int packedStride = K / 2;
        int scalesPerRow = (K + 63) / 64;
        long scalesOffset = (long) N * packedStride;

        // out[M, N] = input[M, K] × W[K, N]^T  +  scale * (input × A^T) × B^T
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                // Base weight contribution
                float acc = 0f;
                for (int k = 0; k < K; k++) {
                    float w;
                    if (packedNf4) {
                        int packedIndex = n * packedStride + (k / 2);
                        byte packed = wNf4.getAtIndex(ValueLayout.JAVA_BYTE, packedIndex);
                        int nibble = (k % 2 == 0) ? ((packed >> 4) & 0xF) : (packed & 0xF);
                        int scaleIdx = (k / 64);
                        short halfBits = wNf4.getAtIndex(ValueLayout.JAVA_SHORT,
                                (scalesOffset / 2) + (long) n * scalesPerRow + scaleIdx);
                        float scale = halfToFloat(halfBits);
                        w = NF4_LOOKUP[nibble] * scale;
                    } else {
                        w = getf(wNf4, (long)(n * K + k));
                    }
                    acc += getf(input, (long)(m * K + k)) * w;
                }
                // LoRA: scale * (input × A^T)[m, r] × B^T[r, n]
                float loraAcc = 0f;
                for (int r = 0; r < R; r++) {
                    float inputA = 0f;
                    for (int k = 0; k < K; k++) {
                        inputA += getf(input, (long)(m * K + k))
                                * getf(loraA, (long)(r * K + k)); // A stored [R, K]
                    }
                    loraAcc += inputA * getf(loraB, (long)(n * R + r)); // B stored [N, R]
                }
                setf(out, (long)(m * N + n), acc + loraScale * loraAcc);
            }
        }
        return 0;
    }

    /** Identity — fp32 is already "unpacked". */
    public static int loadNf4(MemorySegment dst, MemorySegment src, int N, int K) {
        int packedStride = K / 2;
        int scalesPerRow = (K + 63) / 64;
        long scalesOffset = (long) N * packedStride;

        for (int n = 0; n < N; n++) {
            for (int block = 0; block < scalesPerRow; block++) {
                int blockStart = block * 64;
                int blockLen = Math.min(64, K - blockStart);
                if (blockLen <= 0) continue;

                float maxAbs = 1e-8f;
                for (int i = 0; i < blockLen; i++) {
                    float v = getf(src, (long)(n * K + blockStart + i));
                    maxAbs = Math.max(maxAbs, Math.abs(v));
                }
                float scale = maxAbs;
                dst.setAtIndex(ValueLayout.JAVA_SHORT,
                        (scalesOffset / 2) + (long) n * scalesPerRow + block,
                        floatToHalf(scale));

                for (int i = 0; i < blockLen; i += 2) {
                    int idx0 = blockStart + i;
                    int idx1 = idx0 + 1;
                    float v0 = getf(src, (long)(n * K + idx0)) / scale;
                    float v1 = (idx1 < K) ? getf(src, (long)(n * K + idx1)) / scale : 0f;

                    byte hi = nearestNf4(v0);
                    byte lo = nearestNf4(v1);
                    int packedIndex = n * packedStride + (idx0 / 2);
                    dst.setAtIndex(ValueLayout.JAVA_BYTE, packedIndex, (byte)((hi << 4) | (lo & 0xF)));
                }
            }
        }
        return 0;
    }

    /** Identity — fp32 adapter treated as unpacked. */
    public static int loadAdapter(MemorySegment dst, MemorySegment src, int rows, int cols) {
        MemorySegment.copy(src, 0L, dst, 0L, Math.min(src.byteSize(), dst.byteSize()));
        return 0;
    }

    private static float getf(MemorySegment s, long i) {
        return s.getAtIndex(ValueLayout.JAVA_FLOAT, i);
    }

    private static void setf(MemorySegment s, long i, float v) {
        s.setAtIndex(ValueLayout.JAVA_FLOAT, i, v);
    }

    private static byte nearestNf4(float v) {
        byte best = 0;
        float bestDist = Float.MAX_VALUE;
        for (byte i = 0; i < 16; i++) {
            float d = Math.abs(v - NF4_LOOKUP[i]);
            if (d < bestDist) {
                bestDist = d;
                best = i;
            }
        }
        return best;
    }

    private static short floatToHalf(float f) {
        int bits = Float.floatToRawIntBits(f);
        int sign = (bits >>> 16) & 0x8000;
        int exp = ((bits >>> 23) & 0xFF) - 127 + 15;
        int mant = bits & 0x7FFFFF;

        if (exp <= 0) {
            if (exp < -10) {
                return (short) sign;
            }
            mant |= 0x800000;
            int shift = 14 - exp;
            int halfMant = mant >> shift;
            if ((mant >> (shift - 1) & 1) != 0) {
                halfMant += 1;
            }
            return (short) (sign | halfMant);
        } else if (exp >= 31) {
            return (short) (sign | 0x7C00);
        } else {
            int halfMant = mant >> 13;
            if ((mant & 0x1000) != 0) {
                halfMant += 1;
            }
            return (short) (sign | (exp << 10) | (halfMant & 0x3FF));
        }
    }

    private static float halfToFloat(short h) {
        int sign = (h >>> 15) & 0x00000001;
        int exp = (h >>> 10) & 0x0000001f;
        int mant = h & 0x000003ff;
        int bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign << 31;
            } else {
                exp = 1;
                while ((mant & 0x00000400) == 0) {
                    mant <<= 1;
                    exp -= 1;
                }
                mant &= ~0x00000400;
                bits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            bits = (sign << 31) | 0x7f800000 | (mant << 13);
        } else {
            bits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
        return Float.intBitsToFloat(bits);
    }
}
