package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Triplet Loss — trains embeddings so that an anchor is closer to a positive
 * sample than to a negative sample by at least a margin.
 *
 * <p>Based on <em>"FaceNet: A Unified Embedding for Face Recognition"</em>
 * (Schroff et al., 2015).
 *
 * <p>Loss formula:
 * <pre>
 *   L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var loss = new TripletLoss(margin = 0.2f);
 * GradTensor l = loss.forward(anchor, positive, negative);
 * }</pre>
 */
public final class TripletLoss {

    private final float margin;

    /**
     * Creates a triplet loss with the given margin.
     *
     * @param margin minimum distance gap between positive and negative pairs (default 0.2)
     */
    public TripletLoss(float margin) { this.margin = margin; }

    /** Creates a triplet loss with default margin of 0.2. */
    public TripletLoss() { this(0.2f); }

    /**
     * Computes the triplet loss over a batch of anchor/positive/negative embeddings.
     *
     * @param anchor   anchor embeddings {@code [N, D]}
     * @param positive positive embeddings {@code [N, D]} (same class as anchor)
     * @param negative negative embeddings {@code [N, D]} (different class)
     * @return scalar mean triplet loss
     */
    public GradTensor forward(GradTensor anchor, GradTensor positive, GradTensor negative) {
        int N = (int) anchor.shape()[0];
        float[] a = anchor.data(), p = positive.data(), n = negative.data();
        int D = (int) anchor.shape()[1];

        float[] losses = new float[N];
        for (int i = 0; i < N; i++) {
            float distAP = 0f, distAN = 0f;
            for (int d = 0; d < D; d++) {
                float dap = a[i*D+d] - p[i*D+d];
                float dan = a[i*D+d] - n[i*D+d];
                distAP += dap * dap;
                distAN += dan * dan;
            }
            losses[i] = Math.max(0f, distAP - distAN + margin);
        }
        return GradTensor.scalar(VectorOps.sum(losses) / N);
    }
}
