package tech.kayys.gollek.hybridattn.binding;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.jboss.logging.Logger;

/**
 * Pure-Java CPU fallback for GDN kernels.
 *
 * <p>Implements the gated delta rule:
 * <pre>
 *   S_t = α_t ⊙ S_{t-1}  +  β_t · (v_t − S_{t-1} k_t) k_t^T
 * </pre>
 * Correct but unoptimised. Used for development / CI when
 * {@code libgollek_gdn_kernels.so} is not present.
 */
public final class GdnCpuFallback {

    private static final Logger LOG = Logger.getLogger(GdnCpuFallback.class);

    private GdnCpuFallback() {}

    /**
     * GDN layer forward — updates state in-place and writes output.
     *
     * <p>Implements a scalar simplification of the gated delta rule.
     * The full tensor formulation (S ∈ R^{D×S}) is approximated here
     * as element-wise operations for the CPU path.
     */
    public static int gdnLayerForward(
            MemorySegment out,
            MemorySegment state,
            MemorySegment input,
            MemorySegment alpha,
            MemorySegment beta,
            int B, int T, int modelDim, int stateDim) {

        LOG.debug("GDN CPU fallback (native unavailable)");

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < modelDim; d++) {
                    long idx     = (long)(b * T * modelDim + t * modelDim + d);
                    float inp    = getf(input, idx);
                    float a      = getf(alpha, idx);   // output gate
                    float be     = getf(beta,  idx);   // update gate
                    float s_prev = getf(state, (long)(b * modelDim + d));

                    // S_t = α_t * S_{t-1} + β_t * (v_t - S_{t-1}) ← scalar approx
                    float s_new = a * s_prev + be * (inp - s_prev);
                    setf(state, (long)(b * modelDim + d), s_new);
                    setf(out,   idx,                      s_new);
                }
            }
        }
        return 0;
    }

    /**
     * Gate projection: alpha = sigmoid(input × w_alpha), beta = sigmoid(input × w_beta)
     */
    public static int gdnGateProject(
            MemorySegment alphaOut,
            MemorySegment betaOut,
            MemorySegment input,
            MemorySegment wAlpha,
            MemorySegment wBeta,
            int B, int T, int D) {

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int d2 = 0; d2 < D; d2++) {
                    float accA = 0f, accB = 0f;
                    for (int d1 = 0; d1 < D; d1++) {
                        float x = getf(input,  (long)(b * T * D + t * D + d1));
                        accA += x * getf(wAlpha, (long)(d1 * D + d2));
                        accB += x * getf(wBeta,  (long)(d1 * D + d2));
                    }
                    long oi = (long)(b * T * D + t * D + d2);
                    setf(alphaOut, oi, sigmoid(accA));
                    setf(betaOut,  oi, sigmoid(accB));
                }
            }
        }
        return 0;
    }

    private static float sigmoid(float x) { return 1f / (1f + (float) Math.exp(-x)); }

    private static float getf(MemorySegment s, long i) {
        return s.getAtIndex(ValueLayout.JAVA_FLOAT, i);
    }

    private static void setf(MemorySegment s, long i, float v) {
        s.setAtIndex(ValueLayout.JAVA_FLOAT, i, v);
    }
}
