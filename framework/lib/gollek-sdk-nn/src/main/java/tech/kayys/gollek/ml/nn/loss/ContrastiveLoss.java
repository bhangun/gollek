package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Contrastive Loss — pulls together positive pairs and pushes apart negative pairs.
 *
 * <p>Used in self-supervised learning (SimCLR, MoCo, CLIP) and metric learning.
 *
 * <p>Formula:
 * <pre>
 *   L = y · d² + (1-y) · max(0, margin - d)²
 *   where d = ||f(x₁) - f(x₂)||₂
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var loss = new ContrastiveLoss(margin = 1.0f);
 * // y=1 → same class (pull together), y=0 → different class (push apart)
 * GradTensor l = loss.forward(emb1, emb2, labels);
 * }</pre>
 */
public final class ContrastiveLoss {

    private final float margin;

    /** Creates a contrastive loss with default margin of 1.0. */
    public ContrastiveLoss() { this(1.0f); }

    /**
     * Creates a contrastive loss with custom margin.
     *
     * @param margin minimum distance for negative pairs (default 1.0)
     */
    public ContrastiveLoss(float margin) { this.margin = margin; }

    /**
     * Computes the contrastive loss over a batch of pairs.
     *
     * @param x1     first embeddings {@code [N, D]}
     * @param x2     second embeddings {@code [N, D]}
     * @param labels similarity labels {@code [N]}: 1=same class, 0=different class
     * @return scalar mean contrastive loss
     */
    public GradTensor forward(GradTensor x1, GradTensor x2, GradTensor labels) {
        int N = (int) x1.shape()[0], D = (int) x1.shape()[1];
        float[] a = x1.data(), b = x2.data(), y = labels.data();
        float[] losses = new float[N];

        for (int n = 0; n < N; n++) {
            float dist = 0f;
            for (int d = 0; d < D; d++) { float diff = a[n*D+d] - b[n*D+d]; dist += diff*diff; }
            dist = (float) Math.sqrt(dist);
            float label = y[n];
            losses[n] = label * dist * dist
                      + (1f - label) * Math.max(0f, margin - dist) * Math.max(0f, margin - dist);
        }
        return GradTensor.scalar(VectorOps.sum(losses) / N);
    }
}
