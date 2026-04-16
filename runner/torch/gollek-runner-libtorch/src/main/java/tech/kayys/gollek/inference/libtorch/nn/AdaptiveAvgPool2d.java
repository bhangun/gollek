package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.runtime.tensor.Tensor;

/**
 * Adaptive average pooling 2D module.
 * Mirrors {@code libtorch::nn::AdaptiveAvgPool2d}.
 */
public class AdaptiveAvgPool2d extends Module {

    private final long outputH;
    private final long outputW;

    /**
     * @param outputH target output height
     * @param outputW target output width
     */
    public AdaptiveAvgPool2d(long outputH, long outputW) {
        this.outputH = outputH;
        this.outputW = outputW;
    }

    /** Square output size. */
    public AdaptiveAvgPool2d(long outputSize) {
        this(outputSize, outputSize);
    }

    @Override
    public TorchTensor forward(TorchTensor input) {
        return Functional.adaptiveAvgPool2d(input, outputH, outputW);
    }

    @Override
    public String toString() {
        return String.format("AdaptiveAvgPool2d(output_size=(%d, %d))", outputH, outputW);
    }
}
