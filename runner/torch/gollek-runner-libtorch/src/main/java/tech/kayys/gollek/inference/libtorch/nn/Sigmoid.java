package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.runtime.tensor.Tensor;

/**
 * Sigmoid activation module wrapper.
 * Equivalent to {@code libtorch::nn::Sigmoid}.
 */
public class Sigmoid extends Module {

    @Override
    public TorchTensor forward(TorchTensor input) {
        return Functional.sigmoid(input);
    }

    @Override
    public String toString() {
        return "Sigmoid()";
    }
}
