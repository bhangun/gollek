package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.Tensor;

/**
 * Tanh activation module wrapper.
 * Equivalent to {@code torch::nn::Tanh}.
 */
public class Tanh extends Module {

    @Override
    public Tensor forward(Tensor input) {
        return Functional.tanh(input);
    }

    @Override
    public String toString() {
        return "Tanh()";
    }
}
