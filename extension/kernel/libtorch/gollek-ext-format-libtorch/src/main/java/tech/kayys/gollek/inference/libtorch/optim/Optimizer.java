package tech.kayys.gollek.inference.libtorch.optim;

import tech.kayys.gollek.inference.libtorch.core.Tensor;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Abstract base class for all optimizers.
 * Mirrors {@code torch::optim::Optimizer}.
 */
public abstract class Optimizer {

    protected final List<Tensor> parameters;
    protected final double learningRate;

    /**
     * @param parameters   model parameters to optimize
     * @param learningRate base learning rate
     */
    protected Optimizer(List<Tensor> parameters, double learningRate) {
        this.parameters = Collections.unmodifiableList(
                Objects.requireNonNull(parameters, "parameters must not be null"));
        this.learningRate = learningRate;
    }

    /**
     * Perform a single optimization step (parameter update).
     */
    public abstract void step();

    /**
     * Zero all parameter gradients.
     */
    public void zeroGrad() {
        // Gradient zeroing is handled at the native level
        // This is a placeholder for the FFM call
    }

    /**
     * Get the current learning rate.
     */
    public double getLearningRate() {
        return learningRate;
    }
}
