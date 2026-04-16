package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.runtime.tensor.Tensor;

/**
 * ReLU activation module wrapper.
 * Equivalent to {@code libtorch::nn::ReLU}.
 */
public class ReLU extends Module {

    private final boolean inplace;

    public ReLU(boolean inplace) {
        this.inplace = inplace;
    }

    public ReLU() {
        this(false);
    }

    @Override
    public TorchTensor forward(TorchTensor input) {
        return Functional.relu(input);
    }

    @Override
    public String toString() {
        return "ReLU(inplace=" + inplace + ")";
    }
}
