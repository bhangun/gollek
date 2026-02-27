package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.Device;
import tech.kayys.gollek.inference.libtorch.core.ScalarType;
import tech.kayys.gollek.inference.libtorch.core.Tensor;

/**
 * 2D convolution layer.
 * Mirrors {@code torch::nn::Conv2d}.
 */
public class Conv2d extends Module {

    private final long inChannels;
    private final long outChannels;
    private final long kernelSize;
    private final long stride;
    private final long padding;
    private final long dilation;
    private final long groups;
    private final boolean hasBias;

    /**
     * Create a Conv2d layer.
     */
    public Conv2d(long inChannels, long outChannels, long kernelSize,
            long stride, long padding, long dilation, long groups, boolean bias) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.groups = groups;
        this.hasBias = bias;

        // Weight shape: [outChannels, inChannels/groups, kernelSize, kernelSize]
        Tensor weight = Tensor.randn(
                new long[] { outChannels, inChannels / groups, kernelSize, kernelSize },
                ScalarType.FLOAT, Device.CPU);
        registerParameter("weight", weight);

        if (bias) {
            Tensor biasParam = Tensor.zeros(new long[] { outChannels }, ScalarType.FLOAT, Device.CPU);
            registerParameter("bias", biasParam);
        }
    }

    /** Convenience constructor with common defaults. */
    public Conv2d(long inChannels, long outChannels, long kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0, 1, 1, true);
    }

    /** Convenience constructor with stride and padding. */
    public Conv2d(long inChannels, long outChannels, long kernelSize, long stride, long padding) {
        this(inChannels, outChannels, kernelSize, stride, padding, 1, 1, true);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor weight = parameters.get("weight");
        Tensor bias = hasBias ? parameters.get("bias") : null;
        return Functional.conv2d(input, weight, bias, stride, padding, dilation, groups);
    }

    @Override
    public String toString() {
        return String.format("Conv2d(%d, %d, kernel_size=%d, stride=%d, padding=%d)",
                inChannels, outChannels, kernelSize, stride, padding);
    }
}
