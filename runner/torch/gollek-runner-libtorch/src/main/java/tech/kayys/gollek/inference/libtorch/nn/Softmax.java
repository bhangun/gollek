package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.runtime.tensor.Tensor;

/**
 * Softmax activation module.
 * Equivalent to {@code libtorch::nn::Softmax}.
 */
public class Softmax extends Module {

    private final long dim;

    /**
     * @param dim dimension along which softmax will be computed
     */
    public Softmax(long dim) {
        this.dim = dim;
    }

    /** Default: softmax along last dimension (-1). */
    public Softmax() {
        this(-1);
    }

    @Override
    public TorchTensor forward(TorchTensor input) {
        return Functional.softmax(input, dim);
    }

    @Override
    public String toString() {
        return "Softmax(dim=" + dim + ")";
    }
}
