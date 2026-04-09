package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * A trainable parameter in a neural network module.
 * <p>
 * Parameters are tensors with {@code requiresGrad=true} that
 * are registered in a {@link Module} and updated by optimizers
 * during training. Each parameter tracks its own gradient for
 * backpropagation.
 * <p>
 * Equivalent to {@code torch.nn.Parameter} in PyTorch.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var weight = new Parameter(GradTensor.randn(10, 5));
 * // Gradients accumulate during backpropagation
 * optimizer.step();  // update weight using accumulated gradient
 * weight.zeroGrad();  // reset gradient for next iteration
 * }</pre>
 */
public final class Parameter {

    private GradTensor data;

    /**
     * Create a parameter from a tensor.
     * The tensor's requiresGrad flag is automatically set to true.
     *
     * @param data the underlying tensor (will have requiresGrad=true)
     *
     * @throws IllegalArgumentException if data is null
     */
    public Parameter(GradTensor data) {
        if (data == null) {
            throw new IllegalArgumentException("Parameter data cannot be null");
        }
        this.data = data.requiresGrad(true);
    }

    /**
     * Get the underlying tensor.
     *
     * @return the GradTensor holding the parameter values
     */
    public GradTensor data() {
        return data;
    }

    /**
     * Replace the parameter data (typically used by optimizers for in-place updates).
     *
     * @param newData new tensor data (will have requiresGrad=true)
     *
     * @throws IllegalArgumentException if newData is null
     */
    public void setData(GradTensor newData) {
        if (newData == null) {
            throw new IllegalArgumentException("Parameter data cannot be null");
        }
        this.data = newData.requiresGrad(true);
    }

    /**
     * Get the gradient tensor.
     *
     * @return the gradient tensor from the underlying data
     */
    public GradTensor grad() {
        return data.grad();
    }

    /**
     * Zero the gradient. Call this before each training iteration
     * to avoid accumulating gradients across multiple backward passes.
     */
    public void zeroGrad() {
        data.zeroGrad();
    }

    /**
     * Get the total number of elements in this parameter.
     *
     * @return total number of scalar values (product of all dimensions)
     */
    public long numel() {
        return data.numel();
    }

    @Override
    public String toString() {
        return "Parameter(" + data + ")";
    }
}
