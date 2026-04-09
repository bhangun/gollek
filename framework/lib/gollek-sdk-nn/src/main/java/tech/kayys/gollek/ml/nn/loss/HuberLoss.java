package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Huber Loss (Smooth L1) — quadratic for small errors, linear for large errors.
 *
 * <p>Less sensitive to outliers than MSE while being differentiable everywhere.
 * Used in DQN, object detection (Fast R-CNN), and robust regression.
 *
 * <p>Formula:
 * <pre>
 *   L(y, ŷ) = 0.5·(y-ŷ)²          if |y-ŷ| ≤ δ
 *            = δ·(|y-ŷ| - 0.5·δ)   otherwise
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var loss = new HuberLoss(delta = 1.0f);
 * GradTensor l = loss.forward(pred, target);
 * }</pre>
 */
public final class HuberLoss {

    private final float delta;

    /** Creates a Huber loss with default δ=1.0. */
    public HuberLoss() { this(1.0f); }

    /**
     * Creates a Huber loss with custom δ.
     *
     * @param delta threshold between quadratic and linear regions (default 1.0)
     */
    public HuberLoss(float delta) { this.delta = delta; }

    /**
     * Computes the mean Huber loss.
     *
     * @param pred   predictions (any shape)
     * @param target ground-truth values (same shape)
     * @return scalar mean Huber loss
     */
    public GradTensor forward(GradTensor pred, GradTensor target) {
        float[] p = pred.data(), t = target.data();
        float[] losses = new float[p.length];
        for (int i = 0; i < p.length; i++) {
            float diff = Math.abs(p[i] - t[i]);
            losses[i] = diff <= delta
                ? 0.5f * diff * diff
                : delta * (diff - 0.5f * delta);
        }
        return GradTensor.scalar(VectorOps.sum(losses) / p.length);
    }
}
