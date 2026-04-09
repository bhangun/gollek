package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * Mean Squared Error loss for regression tasks.
 * <p>
 * MSE is one of the most commonly used loss functions for regression problems.
 * It penalizes large errors quadratically, making it sensitive to outliers.
 * {@code MSE = mean((predictions - targets)²)}
 * <p>
 * Equivalent to {@code torch.nn.MSELoss}.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * Loss = (1/n) Σ (y_pred - y_true)²
 * </pre>
 *
 * <h3>Gradient</h3>
 * <pre>
 * dLoss/dy_pred = (2/n) * (y_pred - y_true)
 * </pre>
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Predictions:</b> arbitrary shape</li>
 *   <li><b>Targets:</b> same shape as predictions</li>
 *   <li><b>Output:</b> scalar loss value</li>
 * </ul>
 *
 * <h3>Example: Simple Regression</h3>
 * <pre>{@code
 * var loss = new MSELoss();
 * var predictions = GradTensor.of(new float[]{2.5f, 3.1f, 1.9f}, 3);
 * var targets = GradTensor.of(new float[]{2.0f, 3.0f, 2.0f}, 3);
 *
 * var lossValue = loss.compute(predictions, targets);
 * System.out.println("Loss: " + lossValue.item());  // ~0.01
 * }</pre>
 *
 * <h3>Example: With Backpropagation</h3>
 * <pre>{@code
 * var model = ...;  // your neural network
 * var loss = new MSELoss();
 *
 * // Forward pass
 * var predictions = model.forward(input);
 * var lossValue = loss.compute(predictions, targets);
 *
 * // Backward pass
 * lossValue.backward();
 *
 * // Update parameters
 * optimizer.step();
 * }</pre>
 *
 * <h3>Characteristics</h3>
 * <ul>
 *   <li>Smooth and differentiable everywhere</li>
 *   <li>Penalizes large errors heavily (quadratic penalty)</li>
 *   <li>Sensitive to outliers</li>
 *   <li>Good for models with Gaussian error distributions</li>
 * </ul>
 *
 * <h3>When to Use</h3>
 * <ul>
 *   <li>Continuous regression problems (house prices, temperature prediction, etc.)</li>
 *   <li>When error magnitude is important</li>
 *   <li>When data has few outliers</li>
 * </ul>
 *
 * <h3>Alternatives</h3>
 * <ul>
 *   <li><b>MAE (L1Loss):</b> More robust to outliers</li>
 *   <li><b>SmoothL1Loss:</b> Hybrid between MSE and MAE</li>
 *   <li><b>HuberLoss:</b> Smooth version of L1/L2 hybrid</li>
 * </ul>
 */
public class MSELoss {

    /**
     * Compute mean squared error loss.
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
            totalLoss += diff * diff;
        }
        float meanLoss = totalLoss / n;

        GradTensor out = GradTensor.scalar(meanLoss);
        if (predictions.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("MSELoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() * 2.0f / n;
                    float[] grad = new float[n];
                    for (int i = 0; i < n; i++) {
                        grad[i] = (pData[i] - tData[i]) * scale;
                    }
                    predictions.backward(GradTensor.of(grad, predictions.shape()));
                }
            });
        }
        return out;
    }

    @Override
    public String toString() {
        return "MSELoss()";
    }
}
