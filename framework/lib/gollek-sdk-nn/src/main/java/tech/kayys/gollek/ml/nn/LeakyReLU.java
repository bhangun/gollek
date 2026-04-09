package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * Leaky Rectified Linear Unit (LeakyReLU) activation function.
 * <p>
 * A variant of ReLU that allows a small negative slope for negative inputs,
 * preventing "dying ReLU" problem where neurons become inactive (always output zero).
 * <p>
 * {@code LeakyReLU(x) = x if x > 0, else α*x} where α is a small constant (typically 0.01).
 * <p>
 * Equivalent to {@code torch.nn.LeakyReLU}.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * LeakyReLU(x) = {
 *   x,      if x > 0
 *   α*x,    otherwise
 * }
 * </pre>
 *
 * <h3>Gradient</h3>
 * <pre>
 * dLeakyReLU/dx = {
 *   1,      if x > 0
 *   α,      otherwise
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
 * var leaky = new LeakyReLU(0.01f);  // slope = 0.01
 * var output = leaky.forward(input);
 * }</pre>
 *
 * <h3>Common Slopes</h3>
 * <ul>
 *   <li>Standard: α = 0.01 (original LeakyReLU)</li>
 *   <li>Aggressive: α = 0.001 (very small slope)</li>
 *   <li>Gentle: α = 0.1 (larger slope)</li>
 * </ul>
 *
 * <h3>Advantages Over ReLU</h3>
 * <ul>
 *   <li>Prevents dying ReLU problem</li>
 *   <li>Better gradient flow for negative inputs</li>
 *   <li>Often slightly better performance than ReLU</li>
 * </ul>
 *
 * <h3>When to Use</h3>
 * <ul>
 *   <li>When you notice neurons dying (outputting 0 for all inputs)</li>
 *   <li>In GANs where gradient flow is critical</li>
 *   <li>Deep networks where dead neurons are problematic</li>
 * </ul>
 */
public class LeakyReLU extends NNModule {

    private final float negativeSlope;

    /**
     * Create a LeakyReLU with default slope of 0.01.
     */
    public LeakyReLU() {
        this(0.01f);
    }

    /**
     * Create a LeakyReLU with specified negative slope.
     *
     * @param negativeSlope slope for negative inputs (typically 0.001-0.1)
     *
     * @throws IllegalArgumentException if negativeSlope is negative or >= 1
     */
    public LeakyReLU(float negativeSlope) {
        if (negativeSlope < 0 || negativeSlope >= 1) {
            throw new IllegalArgumentException(
                "negativeSlope must be in [0, 1), got: " + negativeSlope);
        }
        this.negativeSlope = negativeSlope;
    }

    /**
     * Apply LeakyReLU activation to input.
     *
     * @param input tensor of arbitrary shape
     * @return output tensor with same shape
     */
    @Override
    public GradTensor forward(GradTensor input) {
        float[] data = input.data();
        float[] result = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] > 0 ? data[i] : negativeSlope * data[i];
        }

        GradTensor out = GradTensor.of(result, input.shape());
        if (input.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("LeakyReLU") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] ug = upstream.data();
                    float[] grad = new float[data.length];
                    for (int i = 0; i < data.length; i++) {
                        grad[i] = ug[i] * (data[i] > 0 ? 1 : negativeSlope);
                    }
                    input.backward(GradTensor.of(grad, input.shape()));
                }
            });
        }
        return out;
    }

    /**
     * Get the negative slope value.
     *
     * @return negativeSlope
     */
    public float getNegativeSlope() {
        return negativeSlope;
    }

    @Override
    public String toString() {
        return "LeakyReLU(negative_slope=" + negativeSlope + ")";
    }
}
