package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Transposed 2D Convolution (deconvolution) — upsamples spatial dimensions.
 *
 * <p>Equivalent to {@code torch.nn.ConvTranspose2d}. Used in decoder networks,
 * GANs, and semantic segmentation (U-Net decoder path).
 *
 * <p>Output size: {@code H_out = (H_in - 1) * stride - 2*padding + kernel}
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var deconv = new ConvTranspose2d(64, 32, 4, stride=2, padding=1); // 2× upsample
 * GradTensor out = deconv.forward(x); // [N,64,H,W] → [N,32,2H,2W]
 * }</pre>
 */
public class ConvTranspose2d extends NNModule {

    private final int inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW;
    private final Parameter weight; // [C_in, C_out, kH, kW]
    private final Parameter bias;

    public ConvTranspose2d(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0);
    }

    public ConvTranspose2d(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this(inChannels, outChannels, kernelSize, kernelSize, stride, stride, padding, padding, true);
    }

    public ConvTranspose2d(int inC, int outC, int kH, int kW, int sH, int sW, int pH, int pW, boolean useBias) {
        this.inChannels = inC; this.outChannels = outC;
        this.kernelH = kH; this.kernelW = kW;
        this.strideH = sH; this.strideW = sW;
        this.padH = pH;    this.padW = pW;

        float bound = (float) Math.sqrt(2.0 / (inC * kH * kW));
        this.weight = registerParameter("weight",
            GradTensor.of(randomUniform(inC * outC * kH * kW, -bound, bound), inC, outC, kH, kW));
        this.bias = useBias
            ? registerParameter("bias", GradTensor.of(new float[outC], outC))
            : null;
    }

    /**
     * Forward pass via col2im (transpose of im2col).
     *
     * @param x input {@code [N, C_in, H, W]}
     * @return output {@code [N, C_out, H_out, W_out]}
     */
    @Override
    public GradTensor forward(GradTensor x) {
        long[] s = x.shape();
        int N = (int)s[0], Cin = (int)s[1], H = (int)s[2], W = (int)s[3];
        int Hout = (H - 1) * strideH - 2 * padH + kernelH;
        int Wout = (W - 1) * strideW - 2 * padW + kernelW;

        float[] xd = x.data(), wd = weight.data().data();
        float[] out = new float[N * outChannels * Hout * Wout];

        // Transposed conv = scatter-add of weighted patches
        for (int n = 0; n < N; n++)
            for (int ci = 0; ci < Cin; ci++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++) {
                        float xVal = xd[n*Cin*H*W + ci*H*W + h*W + w];
                        for (int co = 0; co < outChannels; co++)
                            for (int kh = 0; kh < kernelH; kh++)
                                for (int kw = 0; kw < kernelW; kw++) {
                                    int oh = h * strideH - padH + kh;
                                    int ow = w * strideW - padW + kw;
                                    if (oh < 0 || oh >= Hout || ow < 0 || ow >= Wout) continue;
                                    out[n*outChannels*Hout*Wout + co*Hout*Wout + oh*Wout + ow]
                                        += xVal * wd[ci*outChannels*kernelH*kernelW + co*kernelH*kernelW + kh*kernelW + kw];
                                }
                    }

        if (bias != null) {
            float[] bd = bias.data().data();
            for (int n = 0; n < N; n++)
                for (int co = 0; co < outChannels; co++) {
                    int base = n*outChannels*Hout*Wout + co*Hout*Wout;
                    for (int i = base; i < base + Hout*Wout; i++) out[i] += bd[co];
                }
        }
        return GradTensor.of(out, N, outChannels, Hout, Wout);
    }

    private static float[] randomUniform(int n, float lo, float hi) {
        float[] d = new float[n]; java.util.Random rng = new java.util.Random();
        for (int i = 0; i < n; i++) d[i] = lo + rng.nextFloat() * (hi - lo);
        return d;
    }

    @Override public String toString() {
        return String.format("ConvTranspose2d(%d→%d, kernel=(%d,%d), stride=(%d,%d))",
            inChannels, outChannels, kernelH, kernelW, strideH, strideW);
    }
}
