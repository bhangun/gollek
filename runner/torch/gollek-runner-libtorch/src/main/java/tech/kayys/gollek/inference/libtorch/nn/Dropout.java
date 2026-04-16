package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.runtime.tensor.Tensor;

/**
 * Dropout layer for regularization during training.
 */
public class Dropout extends Module {

    private final double probability;

    /**
     * Create a dropout layer.
     *
     * @param p probability of dropping an element (default 0.5)
     */
    public Dropout(double p) {
        this.probability = p;
    }

    public Dropout() {
        this(0.5);
    }

    @Override
    public TorchTensor forward(TorchTensor input) {
        if (training) {
            return Functional.dropout(input, probability, true);
        }
        return input;
    }

    @Override
    public String toString() {
        return String.format("Dropout(p=%.2f)", probability);
    }
}
