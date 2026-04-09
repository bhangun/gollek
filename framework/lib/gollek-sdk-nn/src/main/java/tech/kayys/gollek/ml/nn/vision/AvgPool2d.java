package tech.kayys.gollek.ml.nn.vision;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;

/**
 * Applies a 2D average pooling over an input signal composed of several input planes.
 */
public class AvgPool2d extends NNModule {
    
    private final int kernelSize;
    private final int stride;
    private final int padding;

    public AvgPool2d(int kernelSize) {
        this(kernelSize, kernelSize, 0);
    }

    public AvgPool2d(int kernelSize, int stride, int padding) {
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public GradTensor forward(GradTensor input) {
        long[] shape = input.shape();
        int b = (int) shape[0];
        int c = (int) shape[1];
        int h = (int) shape[2];
        int w = (int) shape[3];

        int outH = (h + 2 * padding - kernelSize) / stride + 1;
        int outW = (w + 2 * padding - kernelSize) / stride + 1;

        float[] output = new float[b * c * outH * outW];
        return GradTensor.of(output, b, c, outH, outW);
    }
}
