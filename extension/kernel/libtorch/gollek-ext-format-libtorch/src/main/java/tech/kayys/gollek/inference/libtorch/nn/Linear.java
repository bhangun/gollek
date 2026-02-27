package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.Device;
import tech.kayys.gollek.inference.libtorch.core.ScalarType;
import tech.kayys.gollek.inference.libtorch.core.Tensor;

/**
 * Fully connected (linear) layer: y = xW^T + b.
 * Mirrors {@code torch::nn::Linear}.
 */
public class Linear extends Module {

    private final long inFeatures;
    private final long outFeatures;
    private final boolean hasBias;

    /**
     * Create a linear layer.
     *
     * @param inFeatures  size of the input feature dimension
     * @param outFeatures size of the output feature dimension
     * @param bias        whether to include a bias term
     */
    public Linear(long inFeatures, long outFeatures, boolean bias) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.hasBias = bias;

        // Initialize weight with Kaiming uniform init (randn as approximation)
        Tensor weight = Tensor.randn(new long[] { outFeatures, inFeatures }, ScalarType.FLOAT, Device.CPU);
        registerParameter("weight", weight);

        if (bias) {
            Tensor biasParam = Tensor.zeros(new long[] { outFeatures }, ScalarType.FLOAT, Device.CPU);
            registerParameter("bias", biasParam);
        }
    }

    /**
     * Create a linear layer with bias.
     */
    public Linear(long inFeatures, long outFeatures) {
        this(inFeatures, outFeatures, true);
    }

    @Override
    public Tensor forward(Tensor input) {
        // y = x @ W^T
        Tensor weight = parameters.get("weight");
        Tensor output = input.matmul(weight.transpose(0, 1));

        if (hasBias) {
            Tensor bias = parameters.get("bias");
            output = output.add(bias);
        }
        return output;
    }

    public long inFeatures() {
        return inFeatures;
    }

    public long outFeatures() {
        return outFeatures;
    }

    @Override
    public String toString() {
        return String.format("Linear(in_features=%d, out_features=%d, bias=%b)", inFeatures, outFeatures, hasBias);
    }
}
