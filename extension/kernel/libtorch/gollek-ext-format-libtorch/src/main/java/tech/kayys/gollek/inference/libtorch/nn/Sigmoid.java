package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.Tensor;

/**
 * Sigmoid activation module wrapper.
 * Equivalent to {@code torch::nn::Sigmoid}.
 */
public class Sigmoid extends Module {

    @Override
    public Tensor forward(Tensor input) {
        return Functional.sigmoid(input);
    }

    @Override
    public String toString() {
        return "Sigmoid()";
    }
}
