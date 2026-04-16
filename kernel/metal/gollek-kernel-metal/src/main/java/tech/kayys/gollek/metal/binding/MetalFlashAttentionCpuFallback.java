package tech.kayys.gollek.metal.binding;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.jboss.logging.Logger;

/**
 * Pure-Java CPU fallback for Metal FlashAttention-4.
 *
 * <p>
 * Standard causal softmax attention in Java. Correct, not optimised.
 * Used when {@code libgollek_metal.dylib} is absent (non-macOS CI).
 */
public final class MetalFlashAttentionCpuFallback {

    private static final Logger LOG = Logger.getLogger(MetalFlashAttentionCpuFallback.class);

    private MetalFlashAttentionCpuFallback() {
    }

    /**
     * Standard softmax attention fallback.
     * K and V are already gathered (contiguous), not paged.
     */
    public static int execute(
            MemorySegment output,
            MemorySegment query,
            MemorySegment key,
            MemorySegment value,
            int B, int T, int S, int H, int Hkv, int D,
            float scale, boolean isCausal) {

        LOG.debug("MetalFlashAttention CPU fallback (native unavailable)");

        int groupSize = H / Hkv;

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                int hk = h / groupSize;

                for (int t = 0; t < T; t++) {
                    float[] scores = new float[S];
                    float maxS = Float.NEGATIVE_INFINITY;

                    for (int s = 0; s < S; s++) {
                        if (isCausal && s > t) {
                            scores[s] = Float.NEGATIVE_INFINITY;
                            continue;
                        }
                        float dot = 0f;
                        for (int d = 0; d < D; d++) {
                            long qi = ((long) (b * T * H + t * H + h) * D + d);
                            long ki = ((long) (b * S * Hkv + s * Hkv + hk) * D + d);
                            dot += getf(query, qi) * getf(key, ki);
                        }
                        scores[s] = dot * scale;
                        if (scores[s] > maxS)
                            maxS = scores[s];
                    }

                    float sum = 0f;
                    for (int s = 0; s < S; s++) {
                        if (!isCausal || s <= t) {
                            scores[s] = (float) Math.exp(scores[s] - maxS);
                            sum += scores[s];
                        } else {
                            scores[s] = 0f;
                        }
                    }
                    if (sum > 0f)
                        for (int s = 0; s < S; s++)
                            scores[s] /= sum;

                    for (int d = 0; d < D; d++) {
                        float acc = 0f;
                        for (int s = 0; s < S; s++) {
                            long vi = ((long) (b * S * Hkv + s * Hkv + hk) * D + d);
                            acc += scores[s] * getf(value, vi);
                        }
                        long oi = ((long) (b * T * H + t * H + h) * D + d);
                        setf(output, oi, acc);
                    }
                }
            }
        }
        return 0;
    }

    private static float getf(MemorySegment s, long i) {
        return s.getAtIndex(ValueLayout.JAVA_FLOAT, i);
    }

    private static void setf(MemorySegment s, long i, float v) {
        s.setAtIndex(ValueLayout.JAVA_FLOAT, i, v);
    }
}
