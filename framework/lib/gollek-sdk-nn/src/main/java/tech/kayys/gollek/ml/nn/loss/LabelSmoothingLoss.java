package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Label Smoothing Cross-Entropy Loss — regularizes classification by
 * softening hard one-hot targets toward a uniform distribution.
 *
 * <p>Prevents overconfident predictions and improves calibration.
 * Used in image classification (ViT, EfficientNet) and NLP (BERT fine-tuning).
 *
 * <p>Smoothed target: {@code y_smooth = (1-ε)·y_hard + ε/C}
 * where {@code ε} is the smoothing factor and {@code C} is the number of classes.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var loss = new LabelSmoothingLoss(smoothing = 0.1f);
 * GradTensor l = loss.forward(logits, labels); // logits [N,C], labels [N]
 * }</pre>
 */
public final class LabelSmoothingLoss {

    private final float smoothing;

    /**
     * Creates a label smoothing loss.
     *
     * @param smoothing smoothing factor ε in [0, 1) (default 0.1)
     */
    public LabelSmoothingLoss(float smoothing) { this.smoothing = smoothing; }

    /** Creates a label smoothing loss with default ε=0.1. */
    public LabelSmoothingLoss() { this(0.1f); }

    /**
     * Computes the label-smoothed cross-entropy loss.
     *
     * @param logits raw model outputs {@code [N, C]}
     * @param labels integer class indices {@code [N]}
     * @return scalar mean loss
     */
    public GradTensor forward(GradTensor logits, GradTensor labels) {
        long[] s = logits.shape();
        int N = (int) s[0], C = (int) s[1];
        float[] lg = logits.data(), lb = labels.data();
        float[] losses = new float[N];

        for (int n = 0; n < N; n++) {
            // Log-softmax for numerical stability
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < C; c++) max = Math.max(max, lg[n*C+c]);
            float sumExp = 0f;
            for (int c = 0; c < C; c++) sumExp += Math.exp(lg[n*C+c] - max);
            float logSumExp = max + (float) Math.log(sumExp);

            int cls = (int) lb[n];
            float hardLoss = logSumExp - lg[n*C+cls]; // CE for true class

            // Uniform loss over all classes (smoothing term)
            float uniformLoss = 0f;
            for (int c = 0; c < C; c++) uniformLoss += logSumExp - lg[n*C+c];
            uniformLoss /= C;

            losses[n] = (1f - smoothing) * hardLoss + smoothing * uniformLoss;
        }
        return GradTensor.scalar(VectorOps.sum(losses) / N);
    }
}
