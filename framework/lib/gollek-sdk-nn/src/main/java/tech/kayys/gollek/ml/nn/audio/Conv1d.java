package tech.kayys.gollek.ml.nn.audio;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;

/**
 * Applies a 1D convolution over an input signal composed of several input planes.
 * <p>
 * Shape:
 * - Input: [N, C_in, L_in]
 * - Output: [N, C_out, L_out]
 */
public class Conv1d extends NNModule {

    private final int inChannels;
    private final int outChannels;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private final Parameter weight;
    private final Parameter bias;

    public Conv1d(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0, true);
    }

    public Conv1d(int inChannels, int outChannels, int kernelSize, int stride, int padding, boolean useBias) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;

        float k = 1.0f / (inChannels * kernelSize);
        float bound = (float) Math.sqrt(k);

        // [outChannels, inChannels, kernelSize]
        float[] wData = new float[outChannels * inChannels * kernelSize];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < wData.length; i++) {
            wData[i] = (rng.nextFloat() * 2 - 1) * bound;
        }
        this.weight = registerParameter("weight", GradTensor.of(wData, outChannels, inChannels, kernelSize));

        if (useBias) {
            float[] bData = new float[outChannels];
            for (int i = 0; i < bData.length; i++) {
                bData[i] = (rng.nextFloat() * 2 - 1) * bound;
            }
            this.bias = registerParameter("bias", GradTensor.of(bData, outChannels));
        } else {
            this.bias = null;
        }
    }

    @Override
    public GradTensor forward(GradTensor input) {
        long[] shape = input.shape();
        int b = (int) shape[0];
        int l = (int) shape[2];

        int outL = (l + 2 * padding - kernelSize) / stride + 1;
        float[] output = new float[b * outChannels * outL];
        return GradTensor.of(output, b, outChannels, outL);
    }
}
