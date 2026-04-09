package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * 2D Max Pooling — equivalent to {@code torch.nn.MaxPool2d}.
 *
 * <p>Input:  {@code [N, C, H, W]}
 * <p>Output: {@code [N, C, H_out, W_out]}
 * where {@code H_out = (H + 2*padding - kernelSize) / stride + 1}
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var pool = new MaxPool2d(2);        // 2x2, stride=2 (halves spatial dims)
 * var pool = new MaxPool2d(3, 1, 1);  // 3x3, stride=1, padding=1 (same size)
 * }</pre>
 */
public class MaxPool2d extends NNModule {

    private final int kernelH;
    private final int kernelW;
    private final int strideH;
    private final int strideW;
    private final int padH;
    private final int padW;

    public MaxPool2d(int kernelSize) {
        this(kernelSize, kernelSize, kernelSize, kernelSize, 0, 0);
    }

    public MaxPool2d(int kernelSize, int stride, int padding) {
        this(kernelSize, kernelSize, stride, stride, padding, padding);
    }

    public MaxPool2d(int kernelH, int kernelW, int strideH, int strideW, int padH, int padW) {
        this.kernelH = kernelH;
        this.kernelW = kernelW;
        this.strideH = strideH;
        this.strideW = strideW;
        this.padH = padH;
        this.padW = padW;
    }

    @Override
    public GradTensor forward(GradTensor input) {
        long[] s = input.shape();
        int N = (int) s[0], C = (int) s[1], H = (int) s[2], W = (int) s[3];
        int Hout = (H + 2 * padH - kernelH) / strideH + 1;
        int Wout = (W + 2 * padW - kernelW) / strideW + 1;

        float[] in  = input.data();
        float[] out = new float[N * C * Hout * Wout];

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int oh = 0; oh < Hout; oh++) {
                    for (int ow = 0; ow < Wout; ow++) {
                        float max = Float.NEGATIVE_INFINITY;
                        for (int kh = 0; kh < kernelH; kh++) {
                            int ih = oh * strideH - padH + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < kernelW; kw++) {
                                int iw = ow * strideW - padW + kw;
                                if (iw < 0 || iw >= W) continue;
                                float v = in[n * C * H * W + c * H * W + ih * W + iw];
                                if (v > max) max = v;
                            }
                        }
                        out[n * C * Hout * Wout + c * Hout * Wout + oh * Wout + ow] = max;
                    }
                }
            }
        }
        return GradTensor.of(out, N, C, Hout, Wout);
    }

    @Override
    public String toString() {
        return String.format("MaxPool2d(kernel=(%d,%d), stride=(%d,%d), padding=(%d,%d))",
            kernelH, kernelW, strideH, strideW, padH, padW);
    }
}
