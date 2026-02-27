package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.Tensor;

/**
 * GELU activation module wrapper.
 * Equivalent to {@code torch::nn::GELU}.
 */
public class GELU extends Module {

    @Override
    public Tensor forward(Tensor input) {
        return Functional.gelu(input);
    }

    @Override
    public String toString() {
        return "GELU()";
    }
}
