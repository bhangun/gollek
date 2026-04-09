package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * 2D Average Pooling — equivalent to {@code torch.nn.AvgPool2d}.
 * Input: {@code [N, C, H, W]} → Output: {@code [N, C, H_out, W_out]}
 */
public class AvgPool2d extends NNModule {

    private final int kernelH, kernelW, strideH, strideW, padH, padW;

    public AvgPool2d(int kernelSize) {
        this(kernelSize, kernelSize, kernelSize, kernelSize, 0, 0);
    }

    public AvgPool2d(int kernelSize, int stride, int padding) {
        this(kernelSize, kernelSize, stride, stride, padding, padding);
    }

    public AvgPool2d(int kH, int kW, int sH, int sW, int pH, int pW) {
        this.kernelH = kH; this.kernelW = kW;
        this.strideH = sH; this.strideW = sW;
        this.padH = pH;    this.padW = pW;
    }

    @Override
    public GradTensor forward(GradTensor input) {
        long[] s = input.shape();
        int N = (int)s[0], C = (int)s[1], H = (int)s[2], W = (int)s[3];
        int Hout = (H + 2*padH - kernelH) / strideH + 1;
        int Wout = (W + 2*padW - kernelW) / strideW + 1;
        float kArea = kernelH * kernelW;

        float[] in  = input.data();
        float[] out = new float[N * C * Hout * Wout];

        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int oh = 0; oh < Hout; oh++)
                    for (int ow = 0; ow < Wout; ow++) {
                        float sum = 0; int count = 0;
                        for (int kh = 0; kh < kernelH; kh++) {
                            int ih = oh * strideH - padH + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < kernelW; kw++) {
                                int iw = ow * strideW - padW + kw;
                                if (iw < 0 || iw >= W) continue;
                                sum += in[n*C*H*W + c*H*W + ih*W + iw];
                                count++;
                            }
                        }
                        out[n*C*Hout*Wout + c*Hout*Wout + oh*Wout + ow] = count > 0 ? sum / count : 0;
                    }

        GradTensor result = GradTensor.of(out, N, C, Hout, Wout);
        if (input.requiresGrad()) {
            result.requiresGrad(true);
            result.setGradFn(new tech.kayys.gollek.ml.autograd.Function.Context("AvgPool2dBackward") {
                @Override public void backward(GradTensor g) {
                    float[] dIn = new float[N * C * H * W];
                    float[] gd = g.data();
                    for (int n = 0; n < N; n++)
                        for (int c = 0; c < C; c++)
                            for (int oh = 0; oh < Hout; oh++)
                                for (int ow = 0; ow < Wout; ow++) {
                                    float grad = gd[n*C*Hout*Wout + c*Hout*Wout + oh*Wout + ow];
                                    int count = 0;
                                    for (int kh = 0; kh < kernelH; kh++) { int ih = oh*strideH-padH+kh; if (ih>=0&&ih<H) for (int kw=0;kw<kernelW;kw++){int iw=ow*strideW-padW+kw;if(iw>=0&&iw<W)count++;}}
                                    if (count == 0) continue;
                                    float dVal = grad / count;
                                    for (int kh = 0; kh < kernelH; kh++) {
                                        int ih = oh*strideH-padH+kh; if (ih<0||ih>=H) continue;
                                        for (int kw = 0; kw < kernelW; kw++) {
                                            int iw = ow*strideW-padW+kw; if (iw<0||iw>=W) continue;
                                            dIn[n*C*H*W + c*H*W + ih*W + iw] += dVal;
                                        }
                                    }
                                }
                    input.backward(GradTensor.of(dIn, N, C, H, W));
                }
            });
        }
        return result;
    }

    @Override public String toString() {
        return String.format("AvgPool2d(kernel=(%d,%d), stride=(%d,%d))", kernelH, kernelW, strideH, strideW);
    }
}
