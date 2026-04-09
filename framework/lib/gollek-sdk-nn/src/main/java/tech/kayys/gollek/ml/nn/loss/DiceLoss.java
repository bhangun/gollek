package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Dice Loss — overlap-based loss for image segmentation tasks.
 *
 * <p>Optimizes the Dice coefficient (F1 score over pixels), which handles
 * class imbalance better than cross-entropy for segmentation.
 *
 * <p>Formula:
 * <pre>
 *   Dice = 2 * |A ∩ B| / (|A| + |B|)
 *   L    = 1 - Dice
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var loss = new DiceLoss();
 * GradTensor l = loss.forward(predicted, target); // both [N, H, W] in [0,1]
 * }</pre>
 */
public final class DiceLoss {

    private final float smooth; // smoothing to avoid division by zero

    /**
     * Creates a Dice loss with default smoothing of 1.0.
     */
    public DiceLoss() { this(1.0f); }

    /**
     * Creates a Dice loss with custom smoothing.
     *
     * @param smooth Laplace smoothing constant (default 1.0)
     */
    public DiceLoss(float smooth) { this.smooth = smooth; }

    /**
     * Computes the Dice loss between predicted probabilities and binary targets.
     *
     * @param pred   predicted probabilities {@code [N, ...]} in range [0, 1]
     * @param target binary ground-truth mask {@code [N, ...]} with values 0 or 1
     * @return scalar Dice loss in range [0, 1]
     */
    public GradTensor forward(GradTensor pred, GradTensor target) {
        float[] p = pred.data(), t = target.data();

        // intersection = sum(p * t), union = sum(p) + sum(t)
        float[] pt = new float[p.length];
        VectorOps.mul(p, t, pt);
        float intersection = VectorOps.sum(pt);
        float sumP = VectorOps.sum(p);
        float sumT = VectorOps.sum(t);

        float dice = (2f * intersection + smooth) / (sumP + sumT + smooth);
        return GradTensor.scalar(1f - dice);
    }
}
