package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.runtime.tensor.Tensor;

/**
 * Tanh activation module wrapper.
 * Equivalent to {@code libtorch::nn::Tanh}.
 */
public class Tanh extends Module {

    @Override
    public TorchTensor forward(TorchTensor input) {
        return Functional.tanh(input);
    }

    @Override
    public String toString() {
        return "Tanh()";
    }
}
