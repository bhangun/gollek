package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Rectified Linear Unit (ReLU) activation function.
 * <p>
 * Applies the element-wise function: {@code y = max(0, x)}.
 * <p>
 * ReLU is the most commonly used activation function in deep learning
 * due to its computational efficiency and effectiveness at preventing
 * vanishing gradients. It introduces non-linearity while being simple to compute.
 * <p>
 * Equivalent to {@code torch.nn.ReLU}.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * ReLU(x) = {
 *   x,  if x > 0
 *   0,  otherwise
 * }
 * </pre>
 *
 * <h3>Gradient</h3>
 * <pre>
 * dReLU/dx = {
 *   1,  if x > 0
 *   0,  otherwise
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
 * var relu = new ReLU();
 * var output = relu.forward(input);  // Shape unchanged
 * }</pre>
 *
 * <h3>Typical Usage</h3>
 * <pre>{@code
 * var model = new Sequential(
 *     new Linear(784, 256),
 *     new ReLU(),  // Add non-linearity
 *     new Linear(256, 10)
 * );
 * }</pre>
 */
public class ReLU extends NNModule {

    /**
     * Create a ReLU activation function.
     */
    public ReLU() {
    }

    /**
     * Apply ReLU activation to input.
     *
     * @param input tensor of arbitrary shape
     * @return output tensor with same shape, zero-clipped
     */
    @Override
    public GradTensor forward(GradTensor input) {
        return input.relu();
    }

    @Override
    public String toString() {
        return "ReLU()";
    }
}
