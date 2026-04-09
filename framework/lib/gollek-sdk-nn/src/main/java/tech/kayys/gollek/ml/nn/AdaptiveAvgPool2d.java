package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Adaptive Average Pooling — pools to a fixed output size regardless of input size.
 *
 * <p>Equivalent to {@code torch.nn.AdaptiveAvgPool2d}. Used as the final pooling
 * layer in ResNet, EfficientNet, and other classification networks.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var pool = new AdaptiveAvgPool2d(1, 1); // global average pooling → [N, C, 1, 1]
 * var pool = new AdaptiveAvgPool2d(7, 7); // pool to 7×7 regardless of input size
 * }</pre>
 */
public class AdaptiveAvgPool2d extends NNModule {

    private final int outH, outW;

    /**
     * Creates an adaptive average pooling layer.
     *
     * @param outH target output height
     * @param outW target output width
     */
    public AdaptiveAvgPool2d(int outH, int outW) {
        this.outH = outH;
        this.outW = outW;
    }

    /** Creates a global average pooling layer (output size 1×1). */
    public AdaptiveAvgPool2d(int outputSize) { this(outputSize, outputSize); }

    @Override
    public GradTensor forward(GradTensor x) {
        long[] s = x.shape();
        int N = (int)s[0], C = (int)s[1], H = (int)s[2], W = (int)s[3];
        float[] d = x.data();
        float[] out = new float[N * C * outH * outW];

        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++) {
                        // Map output position to input range
                        int hStart = oh * H / outH, hEnd = (oh + 1) * H / outH;
                        int wStart = ow * W / outW, wEnd = (ow + 1) * W / outW;
                        float sum = 0; int count = 0;
                        for (int ih = hStart; ih < hEnd; ih++)
                            for (int iw = wStart; iw < wEnd; iw++) {
                                sum += d[n*C*H*W + c*H*W + ih*W + iw];
                                count++;
                            }
                        out[n*C*outH*outW + c*outH*outW + oh*outW + ow] = count > 0 ? sum/count : 0;
                    }
        return GradTensor.of(out, N, C, outH, outW);
    }

    @Override public String toString() {
        return "AdaptiveAvgPool2d(output=(" + outH + "," + outW + "))";
    }
}
