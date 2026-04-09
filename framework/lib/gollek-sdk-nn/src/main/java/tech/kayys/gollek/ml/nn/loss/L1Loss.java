package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * L1 Loss (Mean Absolute Error) for regression tasks.
 * <p>
 * L1 loss measures the absolute difference between predictions and targets.
 * It is more robust to outliers than MSE because it increases linearly rather than quadratically.
 * <p>
 * {@code L1 = mean(|predictions - targets|)}
 * <p>
 * Equivalent to {@code torch.nn.L1Loss}.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * Loss = (1/n) Σ |y_pred - y_true|
 * </pre>
 *
 * <h3>Gradient (Subgradient)</h3>
 * <pre>
 * dLoss/dy_pred = (1/n) * sign(y_pred - y_true)
 *
 * Note: Not technically differentiable at y_pred = y_true,
 * but we use 0 as the gradient at that point (or sign(0) = 0).
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
 * var loss = new L1Loss();
 * var predictions = GradTensor.of(new float[]{2.5f, 3.1f, 1.9f}, 3);
 * var targets = GradTensor.of(new float[]{2.0f, 3.0f, 2.0f}, 3);
 *
 * // L1 = (|0.5| + |0.1| + |-0.1|) / 3 = 0.7 / 3 ≈ 0.233
 * var lossValue = loss.compute(predictions, targets);
 * }</pre>
 *
 * <h3>Characteristics</h3>
 * <ul>
 *   <li>Robust to outliers (linear rather than quadratic penalty)</li>
 *   <li>Less smooth than MSE (subgradient at zero)</li>
 *   <li>Encourages sparse solutions</li>
 *   <li>Good for data with occasional extreme values</li>
 * </ul>
 *
 * <h3>When to Use</h3>
 * <ul>
 *   <li>Regression with outliers in data</li>
 *   <li>When you want to penalize large errors less severely</li>
 *   <li>Robust regression problems</li>
 *   <li>Alternatives: MSELoss, SmoothL1Loss, HuberLoss</li>
 * </ul>
 *
 * <h3>Comparison to Other Losses</h3>
 * <ul>
 *   <li>vs MSE: L1 is more robust to outliers</li>
 *   <li>vs SmoothL1: SmoothL1 is differentiable everywhere</li>
 *   <li>vs HuberLoss: Huber combines L1 and MSE smoothly</li>
 * </ul>
 */
public class L1Loss {

    /**
     * Compute L1 loss (Mean Absolute Error).
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
            totalLoss += Math.abs(pData[i] - tData[i]);
        }
        float meanLoss = totalLoss / n;

        GradTensor out = GradTensor.scalar(meanLoss);
        if (predictions.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("L1Loss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / n;
                    float[] grad = new float[n];
                    for (int i = 0; i < n; i++) {
                        float diff = pData[i] - tData[i];
                        // sign function: 1 if positive, -1 if negative, 0 if zero
                        grad[i] = diff == 0 ? 0 : (diff > 0 ? 1 : -1);
                        grad[i] *= scale;
                    }
                    predictions.backward(GradTensor.of(grad, predictions.shape()));
                }
            });
        }
        return out;
    }

    @Override
    public String toString() {
        return "L1Loss()";
    }
}
