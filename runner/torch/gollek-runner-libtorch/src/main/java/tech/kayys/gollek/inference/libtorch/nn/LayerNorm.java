package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.Device;
import tech.kayys.gollek.inference.libtorch.core.ScalarType;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.runtime.tensor.Tensor;

/**
 * Layer Normalization.
 * Mirrors {@code libtorch::nn::LayerNorm}.
 */
public class LayerNorm extends Module {

    private final long[] normalizedShape;
    private final double eps;
    private final boolean elementwiseAffine;

    /**
     * Create a LayerNorm layer.
     *
     * @param normalizedShape   shape of the last N dimensions to normalize over
     * @param eps               epsilon for numerical stability
     * @param elementwiseAffine if true, learnable affine parameters
     */
    public LayerNorm(long[] normalizedShape, double eps, boolean elementwiseAffine) {
        this.normalizedShape = normalizedShape;
        this.eps = eps;
        this.elementwiseAffine = elementwiseAffine;

        if (elementwiseAffine) {
            registerParameter("weight", TorchTensor.ones(normalizedShape, ScalarType.FLOAT, Device.CPU));
            registerParameter("bias", TorchTensor.zeros(normalizedShape, ScalarType.FLOAT, Device.CPU));
        }
    }

    /** Convenience constructor with defaults. */
    public LayerNorm(long... normalizedShape) {
        this(normalizedShape, 1e-5, true);
    }

    @Override
    public TorchTensor forward(TorchTensor input) {
        TorchTensor weight = elementwiseAffine ? parameters.get("weight") : null;
        TorchTensor bias = elementwiseAffine ? parameters.get("bias") : null;
        return Functional.layerNorm(input, normalizedShape, weight, bias, eps);
    }

    @Override
    public String toString() {
        return String.format("LayerNorm(%s, eps=%.1e)", java.util.Arrays.toString(normalizedShape), eps);
    }
}
