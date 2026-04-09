package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Focal Loss — down-weights easy examples to focus training on hard ones.
 * Designed for class-imbalanced datasets (e.g. object detection).
 *
 * <p>FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
 *
 * @param gamma focusing parameter (default 2.0) — higher = more focus on hard examples
 * @param alpha class weight (default 0.25)
 */
public class FocalLoss {

    private final float gamma;
    private final float alpha;

    public FocalLoss() { this(2.0f, 0.25f); }

    public FocalLoss(float gamma, float alpha) {
        this.gamma = gamma;
        this.alpha = alpha;
    }

    /**
     * @param logits raw model output [N, C] (before softmax)
     * @param targets class indices [N]
     */
    public GradTensor forward(GradTensor logits, GradTensor targets) {
        long[] shape = logits.shape();
        int N = (int) shape[0], C = (int) shape[1];
        float[] lg = logits.data(), tg = targets.data();

        // Softmax per sample
        float[] probs = new float[N * C];
        for (int n = 0; n < N; n++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < C; c++) max = Math.max(max, lg[n * C + c]);
            float sum = 0;
            for (int c = 0; c < C; c++) { probs[n * C + c] = (float) Math.exp(lg[n * C + c] - max); sum += probs[n * C + c]; }
            for (int c = 0; c < C; c++) probs[n * C + c] /= sum;
        }

        // Focal loss per sample
        float[] losses = new float[N];
        for (int n = 0; n < N; n++) {
            int cls = (int) tg[n];
            float pt = Math.max(probs[n * C + cls], 1e-7f);
            losses[n] = -alpha * (float) Math.pow(1f - pt, gamma) * (float) Math.log(pt);
        }

        float mean = VectorOps.sum(losses) / N;
        return GradTensor.scalar(mean);
    }
}
