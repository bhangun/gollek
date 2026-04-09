package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * Smooth L1 Loss (Huber-like loss) for robust regression.
 * <p>
 * Smooth L1 Loss combines the benefits of L1 and MSE losses. It's quadratic (MSE-like)
 * for small errors and linear (L1-like) for large errors. This makes it less sensitive
 * to outliers than MSE while being fully differentiable everywhere (unlike L1).
 * <p>
 * The transition happens at a threshold (usually 1.0) controlled by the beta parameter.
 * <p>
 * {@code SmoothL1(x) = 0.5*x² if |x| < beta, else beta*(|x| - 0.5*beta)}
 * <p>
 * Equivalent to {@code torch.nn.SmoothL1Loss}.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * For difference d = y_pred - y_true:
 *
 * SmoothL1(d) = {
 *   0.5 * d²,                    if |d| < beta
 *   beta * (|d| - 0.5 * beta),   otherwise
 * }
 *
 * Loss = mean(SmoothL1(differences))
 * </pre>
 *
 * <h3>Gradient</h3>
 * <pre>
 * dSmoothL1/dd = {
 *   d,                if |d| < beta
 *   beta * sign(d),   otherwise
 * }
 * </pre>
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Predictions:</b> arbitrary shape</li>
 *   <li><b>Targets:</b> same shape as predictions</li>
 *   <li><b>Output:</b> scalar loss value</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var loss = new SmoothL1Loss(1.0f);  // beta = 1.0
 * var predictions = GradTensor.of(new float[]{2.5f, 3.1f, 10.0f}, 3);
 * var targets = GradTensor.of(new float[]{2.0f, 3.0f, 2.0f}, 3);
 * var lossValue = loss.compute(predictions, targets);
 * }</pre>
 *
 * <h3>Effect of Beta Parameter</h3>
 * <ul>
 *   <li><b>Small beta (0.1):</b> More aggressive, almost like L1</li>
 *   <li><b>Standard beta (1.0):</b> Balanced robustness and smoothness</li>
 *   <li><b>Large beta (10.0):</b> More like MSE, sensitive to outliers</li>
 * </ul>
 *
 * <h3>Characteristics</h3>
 * <ul>
 *   <li>Smooth everywhere (twice differentiable except at ±beta)</li>
 *   <li>Robust to outliers like L1</li>
 *   <li>Better gradient flow than L1 for small errors</li>
 *   <li>Used in object detection (Faster R-CNN, RetinaNet)</li>
 * </ul>
 *
 * <h3>When to Use</h3>
 * <ul>
 *   <li>Object detection and bounding box regression</li>
 *   <li>Regression with mixed error distributions</li>
 *   <li>When you want robustness without sacrificing smoothness</li>
 *   <li>Generally better than MSE for noisy regression targets</li>
 * </ul>
 *
 * <h3>Common Use Cases</h3>
 * <ul>
 *   <li>Bounding box prediction in object detection</li>
 *   <li>Pose estimation regression</li>
 *   <li>Depth prediction from single images</li>
 * </ul>
 */
public class SmoothL1Loss {

    private final float beta;

    /**
     * Create a SmoothL1Loss with default beta of 1.0.
     */
    public SmoothL1Loss() {
        this(1.0f);
    }

    /**
     * Create a SmoothL1Loss with specified beta threshold.
     *
     * @param beta transition point between quadratic and linear regions (typically 0.5-2.0)
     *
     * @throws IllegalArgumentException if beta is non-positive
     */
    public SmoothL1Loss(float beta) {
        if (beta <= 0) {
            throw new IllegalArgumentException("beta must be positive, got: " + beta);
        }
        this.beta = beta;
    }

    /**
     * Compute Smooth L1 loss.
     *
     * @param predictions predicted values of arbitrary shape
     * @param targets     ground truth values (must have same shape as predictions)
     * @return scalar loss tensor
     *
     * @throws IllegalArgumentException if shapes do not match
     */
    public GradTensor compute(GradTensor predictions, GradTensor targets) {
        long[] pShape = predictions.shape();
        long[] tShape = targets.shape();

        if (!java.util.Arrays.equals(pShape, tShape)) {
            throw new IllegalArgumentException(
                "predictions and targets shapes must match, got: " + java.util.Arrays.toString(pShape) +
                " vs " + java.util.Arrays.toString(tShape));
        }

        float[] pData = predictions.data();
        float[] tData = targets.data();
        int n = pData.length;

        float totalLoss = 0;
        for (int i = 0; i < n; i++) {
            float diff = pData[i] - tData[i];
            float absDiff = Math.abs(diff);
            if (absDiff < beta) {
                totalLoss += 0.5f * diff * diff;
            } else {
                totalLoss += beta * (absDiff - 0.5f * beta);
            }
        }
        float meanLoss = totalLoss / n;

        GradTensor out = GradTensor.scalar(meanLoss);
        if (predictions.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("SmoothL1Loss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / n;
                    float[] grad = new float[n];
                    for (int i = 0; i < n; i++) {
                        float diff = pData[i] - tData[i];
                        float absDiff = Math.abs(diff);
                        if (absDiff < beta) {
                            grad[i] = diff * scale;
                        } else {
                            grad[i] = (diff > 0 ? 1 : -1) * beta * scale;
                        }
                    }
                    predictions.backward(GradTensor.of(grad, predictions.shape()));
                }
            });
        }
        return out;
    }

    /**
     * Get the beta threshold value.
     *
     * @return beta
     */
    public float getBeta() {
        return beta;
    }

    @Override
    public String toString() {
        return "SmoothL1Loss(beta=" + beta + ")";
    }
}
