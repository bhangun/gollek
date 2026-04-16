package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.runtime.tensor.Tensor;

/**
 * GELU activation module wrapper.
 * Equivalent to {@code libtorch::nn::GELU}.
 */
public class GELU extends Module {

    @Override
    public TorchTensor forward(TorchTensor input) {
        return Functional.gelu(input);
    }

    @Override
    public String toString() {
        return "GELU()";
    }
}
