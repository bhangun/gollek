package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.runtime.tensor.Tensor;

/**
 * 2D max pooling module.
 * Mirrors {@code libtorch::nn::MaxPool2d}.
 */
public class MaxPool2d extends Module {

    private final long kernelSize;
    private final long stride;
    private final long padding;

    /**
     * @param kernelSize size of the pooling window
     * @param stride     stride of the pooling window (defaults to kernelSize if 0)
     * @param padding    implicit zero-padding
     */
    public MaxPool2d(long kernelSize, long stride, long padding) {
        this.kernelSize = kernelSize;
        this.stride = stride > 0 ? stride : kernelSize;
        this.padding = padding;
    }

    /** Convenience: stride = kernelSize, padding = 0. */
    public MaxPool2d(long kernelSize) {
        this(kernelSize, kernelSize, 0);
    }

    @Override
    public TorchTensor forward(TorchTensor input) {
        return Functional.maxPool2d(input, kernelSize, stride, padding);
    }

    @Override
    public String toString() {
        return String.format("MaxPool2d(kernel_size=%d, stride=%d, padding=%d)",
                kernelSize, stride, padding);
    }
}
