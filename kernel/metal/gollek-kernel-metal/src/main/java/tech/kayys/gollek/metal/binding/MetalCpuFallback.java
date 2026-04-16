package tech.kayys.gollek.metal.binding;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Pure-Java CPU fallback for MetalBinding.
 *
 * <p>
 * Activated automatically when {@code libgollek_metal.dylib} is absent —
 * identical to the pattern used by {@code FlashAttention3CpuFallback} and
 * {@code PagedAttentionCpuFallback} already in Gollek. Correct but not
 * performance-optimised; intended for development and CI on non-Apple hardware.
 */
final class MetalCpuFallback {

    private MetalCpuFallback() {
    }

    // ── Matrix multiplication ─────────────────────────────────────────────────

    static int matmul(MemorySegment C, MemorySegment A, MemorySegment B,
            int M, int K, int N, float alpha, float beta) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float acc = 0f;
                for (int k = 0; k < K; k++) {
                    acc += get(A, (long) m * K + k) * get(B, (long) k * N + n);
                }
                float prev = beta != 0f ? get(C, (long) m * N + n) : 0f;
                set(C, (long) m * N + n, alpha * acc + prev);
            }
        }
        return 0;
    }

    // ── Softmax attention ─────────────────────────────────────────────────────

    static int attention(MemorySegment out, MemorySegment Q,
            MemorySegment K_cache, MemorySegment V_cache,
            MemorySegment blockTable, MemorySegment contextLens,
            int B, int T, int H, int D,
            int blockSize, int maxBlocks,
            float scale, int isCausal) {
        for (int b = 0; b < B; b++) {
            int ctxLen = contextLens.getAtIndex(ValueLayout.JAVA_INT, b);
            int numBlocks = (ctxLen + blockSize - 1) / blockSize;

            for (int h = 0; h < H; h++) {
                for (int t = 0; t < T; t++) {
                    // Q[b, t, h, :] → scores over all context positions
                    float[] scores = new float[ctxLen];
                    float maxS = Float.NEGATIVE_INFINITY;

                    for (int blk = 0; blk < numBlocks; blk++) {
                        int phys = blockTable.getAtIndex(
                                ValueLayout.JAVA_INT, (long) b * maxBlocks + blk);
                        int tokLen = Math.min(blockSize, ctxLen - blk * blockSize);

                        for (int tok = 0; tok < tokLen; tok++) {
                            int absPos = blk * blockSize + tok;
                            if (isCausal == 1 && absPos > t) {
                                scores[absPos] = -1e9f;
                                continue;
                            }

                            float dot = 0f;
                            for (int d = 0; d < D; d++) {
                                long qOff = ((long) (b * T * H + t * H + h)) * D + d;
                                long kOff = (((long) phys * H + h) * blockSize + tok) * D + d;
                                dot += get(Q, qOff) * get(K_cache, kOff);
                            }
                            dot *= scale;
                            scores[absPos] = dot;
                            if (dot > maxS)
                                maxS = dot;
                        }
                    }

                    // Softmax
                    float sum = 0f;
                    for (int p = 0; p < ctxLen; p++) {
                        scores[p] = (float) Math.exp(scores[p] - maxS);
                        sum += scores[p];
                    }
                    for (int p = 0; p < ctxLen; p++)
                        scores[p] /= sum;

                    // Weighted sum of V
                    for (int d = 0; d < D; d++) {
                        float acc = 0f;
                        for (int blk = 0; blk < numBlocks; blk++) {
                            int phys = blockTable.getAtIndex(
                                    ValueLayout.JAVA_INT, (long) b * maxBlocks + blk);
                            int tokLen = Math.min(blockSize, ctxLen - blk * blockSize);
                            for (int tok = 0; tok < tokLen; tok++) {
                                int absPos = blk * blockSize + tok;
                                long vOff = (((long) phys * H + h) * blockSize + tok) * D + d;
                                acc += scores[absPos] * get(V_cache, vOff);
                            }
                        }
                        long outOff = ((long) (b * T * H + t * H + h)) * D + d;
                        set(out, outOff, acc);
                    }
                }
            }
        }
        return 0;
    }

    // ── RMS Norm ──────────────────────────────────────────────────────────────

    static int rmsNorm(MemorySegment out, MemorySegment x,
            MemorySegment weight, int N, float eps) {
        float ss = 0f;
        for (int i = 0; i < N; i++) {
            float v = get(x, i);
            ss += v * v;
        }
        float inv = 1f / (float) Math.sqrt(ss / N + eps);
        for (int i = 0; i < N; i++)
            set(out, i, get(x, i) * inv * get(weight, i));
        return 0;
    }

    // ── SiLU FFN ──────────────────────────────────────────────────────────────

    static int siluFfn(MemorySegment out, MemorySegment gate,
            MemorySegment up, int N) {
        for (int i = 0; i < N; i++) {
            float g = get(gate, i);
            set(out, i, (g / (1f + (float) Math.exp(-g))) * get(up, i));
        }
        return 0;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static float get(MemorySegment seg, long idx) {
        return seg.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
    }

    private static void set(MemorySegment seg, long idx, float val) {
        seg.setAtIndex(ValueLayout.JAVA_FLOAT, idx, val);
    }
}
