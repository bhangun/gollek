package tech.kayys.gollek.flashattn.binding;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.jboss.logging.Logger;

/**
 * Pure-Java CPU fallback for FlashAttention-4.
 *
 * <p>Functionally equivalent to {@code FlashAttention3CpuFallback} — standard
 * softmax attention in Java. Not suitable for production; used for development
 * and CI on non-Blackwell hardware. Supports GQA (grouped query attention) via
 * the {@code numHeadsK} parameter identical to the FA3 fallback.
 */
public final class FlashAttention4CpuFallback {

    private static final Logger LOG = Logger.getLogger(FlashAttention4CpuFallback.class);

    private FlashAttention4CpuFallback() {}

    public static int execute(
            MemorySegment output,
            MemorySegment query,
            MemorySegment keyCache,
            MemorySegment valueCache,
            int batchSize,
            int seqLen,
            int numHeads,
            int numHeadsK,
            int headDim,
            float softmaxScale,
            boolean isCausal,
            boolean useFp8) {

        LOG.debug("FlashAttention-4 CPU fallback (native unavailable)");
        if (useFp8) LOG.warn("FP8 not supported in CPU fallback — computing in FP32");

        int groupSize = numHeads / numHeadsK;

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                int hk = h / groupSize; // GQA head mapping

                for (int q = 0; q < seqLen; q++) {
                    float[] scores  = new float[seqLen];
                    float   maxScore = Float.NEGATIVE_INFINITY;

                    // QK^T
                    for (int k = 0; k < seqLen; k++) {
                        if (isCausal && k > q) { scores[k] = Float.NEGATIVE_INFINITY; continue; }
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++) {
                            long qi = ((long)(b * seqLen * numHeads  + q  * numHeads  + h)  * headDim + d);
                            long ki = ((long)(b * seqLen * numHeadsK + k  * numHeadsK + hk) * headDim + d);
                            dot += getf(query, qi) * getf(keyCache, ki);
                        }
                        scores[k] = dot * softmaxScale;
                        if (scores[k] > maxScore) maxScore = scores[k];
                    }

                    // Softmax
                    float sum = 0f;
                    for (int k = 0; k < seqLen; k++) {
                        if (!isCausal || k <= q) {
                            scores[k] = (float) Math.exp(scores[k] - maxScore);
                            sum += scores[k];
                        } else {
                            scores[k] = 0f;
                        }
                    }
                    if (sum > 0f) for (int k = 0; k < seqLen; k++) scores[k] /= sum;

                    // Weighted V
                    for (int d = 0; d < headDim; d++) {
                        float acc = 0f;
                        for (int k = 0; k < seqLen; k++) {
                            long vi = ((long)(b * seqLen * numHeadsK + k * numHeadsK + hk) * headDim + d);
                            acc += scores[k] * getf(valueCache, vi);
                        }
                        long oi = ((long)(b * seqLen * numHeads + q * numHeads + h) * headDim + d);
                        setf(output, oi, acc);
                    }
                }
            }
        }
        return 0;
    }

    private static float getf(MemorySegment seg, long idx) {
        return seg.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
    }

    private static void setf(MemorySegment seg, long idx, float v) {
        seg.setAtIndex(ValueLayout.JAVA_FLOAT, idx, v);
    }
}
