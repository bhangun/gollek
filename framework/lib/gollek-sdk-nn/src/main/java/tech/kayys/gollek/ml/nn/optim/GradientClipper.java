package tech.kayys.gollek.ml.nn.optim;

import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.List;

/**
 * Gradient clipping utilities to prevent exploding gradients during training.
 *
 * <p>Two strategies are provided:
 * <ul>
 *   <li>{@link #clipByNorm} — rescales all gradients so the global L2 norm ≤ {@code maxNorm}</li>
 *   <li>{@link #clipByValue} — clamps each gradient element to {@code [minVal, maxVal]}</li>
 * </ul>
 *
 * <p>Clip-by-norm is the standard approach for RNNs and Transformers.
 * Call it <em>before</em> {@code optimizer.step()}.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * loss.backward();
 * float norm = GradientClipper.clipByNorm(model.parameters(), 1.0f);
 * optimizer.step();
 * }</pre>
 */
public final class GradientClipper {

    private GradientClipper() {}

    /**
     * Clips gradients by global L2 norm.
     *
     * <p>Computes the total gradient norm across all parameters:
     * {@code totalNorm = sqrt(Σ ||g||²)}. If {@code totalNorm > maxNorm},
     * scales every gradient by {@code maxNorm / totalNorm}.
     *
     * <p>Uses {@link VectorOps} for SIMD-accelerated inner-product computation.
     *
     * @param params  list of parameters whose gradients to clip
     * @param maxNorm maximum allowed gradient norm (must be positive)
     * @return the pre-clipping total gradient norm
     */
    public static float clipByNorm(List<Parameter> params, float maxNorm) {
        // Compute total squared norm using VectorOps dot-product
        float totalSq = 0f;
        for (Parameter p : params) {
            if (p.data().grad() == null) continue;
            float[] g = p.data().grad().data();
            // sum(g * g) via VectorOps mul then sum
            float[] sq = new float[g.length];
            VectorOps.mul(g, g, sq);
            totalSq += VectorOps.sum(sq);
        }
        float totalNorm = (float) Math.sqrt(totalSq);

        if (totalNorm > maxNorm) {
            float scale = maxNorm / (totalNorm + 1e-6f);
            for (Parameter p : params) {
                if (p.data().grad() == null) continue;
                float[] g = p.data().grad().data();
                VectorOps.mulScalar(g, scale, g);
            }
        }
        return totalNorm;
    }

    /**
     * Clips each gradient element to the range {@code [minVal, maxVal]}.
     *
     * <p>This is a simpler but less principled approach than norm clipping.
     * Useful when gradient distributions are known to be bounded.
     *
     * @param params list of parameters whose gradients to clip
     * @param minVal minimum allowed gradient value
     * @param maxVal maximum allowed gradient value
     */
    public static void clipByValue(List<Parameter> params, float minVal, float maxVal) {
        for (Parameter p : params) {
            if (p.data().grad() == null) continue;
            float[] g = p.data().grad().data();
            for (int i = 0; i < g.length; i++) {
                if (g[i] < minVal) g[i] = minVal;
                else if (g[i] > maxVal) g[i] = maxVal;
            }
        }
    }
}
