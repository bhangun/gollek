package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * Exponential Linear Unit (ELU) activation function.
 * <p>
 * ELU combines the benefits of ReLU and smooth activation functions. For positive inputs,
 * it acts like ReLU; for negative inputs, it saturates smoothly to -α instead of -∞.
 * This helps with vanishing gradient problems and enables self-normalizing networks.
 * <p>
 * {@code ELU(x) = x if x > 0, else α * (exp(x) - 1)}
 * <p>
 * Equivalent to {@code torch.nn.ELU}.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * ELU(x) = {
 *   x,                 if x > 0
 *   α * (exp(x) - 1),  otherwise
 * }
 * </pre>
 *
 * <h3>Gradient</h3>
 * <pre>
 * dELU/dx = {
 *   1,           if x > 0
 *   α * exp(x),  otherwise
 * }
 * </pre>
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Input:</b> arbitrary shape</li>
 *   <li><b>Output:</b> same shape as input</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var elu = new ELU(1.0f);  // α = 1.0
 * var output = elu.forward(input);
 * }</pre>
 *
 * <h3>Common α Values</h3>
 * <ul>
 *   <li>Standard: α = 1.0 (default)</li>
 *   <li>Smaller: α = 0.5 (less saturation)</li>
 *   <li>Larger: α = 2.0 (more saturation)</li>
 * </ul>
 *
 * <h3>Characteristics</h3>
 * <ul>
 *   <li>Smooth activation (differentiable everywhere)</li>
 *   <li>Can output negative values (helps with mean-centering)</li>
 *   <li>Better than ReLU for deep networks</li>
 *   <li>Slightly more computational cost than ReLU</li>
 *   <li>Foundation for SELU (Self-Normalizing Neural Networks)</li>
 * </ul>
 *
 * <h3>When to Use</h3>
 * <ul>
 *   <li>Deep neural networks where mean-centering helps</li>
 *   <li>When you want smoother gradient flow than ReLU</li>
 *   <li>Convolutional networks (often outperforms ReLU)</li>
 *   <li>As alternative to LeakyReLU with more curvature</li>
 * </ul>
 */
public class ELU extends NNModule {

    private final float alpha;

    /**
     * Create an ELU with default alpha of 1.0.
     */
    public ELU() {
        this(1.0f);
    }

    /**
     * Create an ELU with specified alpha.
     *
     * @param alpha saturation point for negative inputs (typically 0.5-2.0)
     *
     * @throws IllegalArgumentException if alpha is non-positive
     */
    public ELU(float alpha) {
        if (alpha <= 0) {
            throw new IllegalArgumentException("alpha must be positive, got: " + alpha);
        }
        this.alpha = alpha;
    }

    /**
     * Apply ELU activation to input.
     *
     * @param input tensor of arbitrary shape
     * @return output tensor with same shape
     */
    @Override
    public GradTensor forward(GradTensor input) {
        float[] data = input.data();
        float[] result = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] > 0 ? data[i] : alpha * ((float) Math.exp(data[i]) - 1);
        }

        GradTensor out = GradTensor.of(result, input.shape());
        if (input.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("ELU") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] ug = upstream.data();
                    float[] grad = new float[data.length];
                    for (int i = 0; i < data.length; i++) {
                        if (data[i] > 0) {
                            grad[i] = ug[i];
                        } else {
                            grad[i] = ug[i] * alpha * (float) Math.exp(data[i]);
                        }
                    }
                    input.backward(GradTensor.of(grad, input.shape()));
                }
            });
        }
        return out;
    }

    /**
     * Get the alpha value.
     *
     * @return alpha
     */
    public float getAlpha() {
        return alpha;
    }

    @Override
    public String toString() {
        return "ELU(alpha=" + alpha + ")";
    }
}
