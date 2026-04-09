package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.Function;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * 2D Convolution layer — equivalent to {@code torch.nn.Conv2d}.
 *
 * <p>Input shape:  {@code [N, C_in,  H, W]}
 * <p>Output shape: {@code [N, C_out, H_out, W_out]}
 *
 * <p>The forward pass uses an im2col + GEMM strategy where the inner
 * GEMM loop is accelerated via the JDK 25 Vector API through
 * {@link VectorOps#matmul}.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var conv = new Conv2d(3, 64, 3);          // 3-channel input, 64 filters, 3x3 kernel
 * var conv = new Conv2d(64, 128, 3, 1, 1);  // stride=1, padding=1 (same)
 * GradTensor out = conv.forward(input);     // [N,3,H,W] → [N,64,H-2,W-2]
 * }</pre>
 */
public class Conv2d extends NNModule {

    private final int inChannels;
    private final int outChannels;
    private final int kernelH;
    private final int kernelW;
    private final int strideH;
    private final int strideW;
    private final int padH;
    private final int padW;

    private final Parameter weight; // [C_out, C_in, kH, kW]
    private final Parameter bias;   // [C_out]

    public Conv2d(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, kernelSize, 1, 1, 0, 0, true);
    }

    public Conv2d(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this(inChannels, outChannels, kernelSize, kernelSize, stride, stride, padding, padding, true);
    }

    public Conv2d(int inChannels, int outChannels, int kernelH, int kernelW,
                  int strideH, int strideW, int padH, int padW, boolean useBias) {
        this.inChannels  = inChannels;
        this.outChannels = outChannels;
        this.kernelH = kernelH;
        this.kernelW = kernelW;
        this.strideH = strideH;
        this.strideW = strideW;
        this.padH = padH;
        this.padW = padW;

        // Kaiming uniform init
        float bound = (float) Math.sqrt(2.0 / (inChannels * kernelH * kernelW));
        int wSize = outChannels * inChannels * kernelH * kernelW;
        float[] wData = randomUniform(wSize, -bound, bound);
        this.weight = registerParameter("weight",
            GradTensor.of(wData, outChannels, inChannels, kernelH, kernelW));

        if (useBias) {
            float[] bData = randomUniform(outChannels, -bound, bound);
            this.bias = registerParameter("bias", GradTensor.of(bData, outChannels));
        } else {
            this.bias = null;
        }
    }

    /**
     * Forward pass: im2col → GEMM (Vector API accelerated).
     *
     * @param input [N, C_in, H, W]
     * @return      [N, C_out, H_out, W_out]
     */
    @Override
    public GradTensor forward(GradTensor input) {
        long[] s = input.shape();
        int N    = (int) s[0];
        int Cin  = (int) s[1];
        int H    = (int) s[2];
        int W    = (int) s[3];

        int Hout = (H + 2 * padH - kernelH) / strideH + 1;
        int Wout = (W + 2 * padW - kernelW) / strideW + 1;

        // im2col: [N, C_in*kH*kW, Hout*Wout]
        int colRows = Cin * kernelH * kernelW;
        int colCols = Hout * Wout;
        float[] col = im2col(input.data(), N, Cin, H, W, Hout, Wout, colRows, colCols);

        // weight reshaped: [C_out, C_in*kH*kW]
        float[] wFlat = weight.data().data();

        // GEMM: [C_out, colRows] x [colRows, colCols] = [C_out, colCols]  per sample
        float[] outData = new float[N * outChannels * Hout * Wout];
        for (int n = 0; n < N; n++) {
            float[] colN = slice(col, n * colRows * colCols, colRows * colCols);
            // VectorOps.matmul uses Vector API internally
            float[] gemm = VectorOps.matmul(wFlat, colN, outChannels, colRows, colCols);
            System.arraycopy(gemm, 0, outData, n * outChannels * Hout * Wout, gemm.length);
        }

        // Add bias
        if (bias != null) {
            float[] bData = bias.data().data();
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < outChannels; c++) {
                    int base = n * outChannels * Hout * Wout + c * Hout * Wout;
                    for (int i = base; i < base + Hout * Wout; i++) {
                        outData[i] += bData[c];
                    }
                }
            }
        }

        GradTensor out = GradTensor.of(outData, N, outChannels, Hout, Wout);

        // Register backward if any input requires grad
        if (input.requiresGrad() || weight.data().requiresGrad()) {
            out.setGradFn(new Conv2dBackward(
                input, weight.data(), bias != null ? bias.data() : null,
                col, N, Cin, H, W, Hout, Wout, colRows, colCols));
        }
        return out;
    }

    // ── im2col ───────────────────────────────────────────────────────────

    private float[] im2col(float[] input, int N, int Cin, int H, int W,
                            int Hout, int Wout, int colRows, int colCols) {
        float[] col = new float[N * colRows * colCols];
        for (int n = 0; n < N; n++) {
            int colBase = n * colRows * colCols;
            int row = 0;
            for (int c = 0; c < Cin; c++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++, row++) {
                        int colIdx = colBase + row * colCols;
                        for (int oh = 0; oh < Hout; oh++) {
                            int ih = oh * strideH - padH + kh;
                            for (int ow = 0; ow < Wout; ow++) {
                                int iw = ow * strideW - padW + kw;
                                float val = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                    ? input[n * Cin * H * W + c * H * W + ih * W + iw]
                                    : 0f;
                                col[colIdx + oh * Wout + ow] = val;
                            }
                        }
                    }
                }
            }
        }
        return col;
    }

    // ── Backward context ─────────────────────────────────────────────────

    private class Conv2dBackward extends tech.kayys.gollek.ml.autograd.Function.Context {
        private final GradTensor input, weightT, biasT;
        private final float[] col; // saved im2col output [N, colRows, colCols]
        private final int N, Cin, H, W, Hout, Wout, colRows, colCols;

        Conv2dBackward(GradTensor input, GradTensor weight, GradTensor bias,
                       float[] col, int N, int Cin, int H, int W,
                       int Hout, int Wout, int colRows, int colCols) {
            super("Conv2dBackward");
            this.input = input; this.weightT = weight; this.biasT = bias;
            this.col = col; this.N = N; this.Cin = Cin; this.H = H; this.W = W;
            this.Hout = Hout; this.Wout = Wout; this.colRows = colRows; this.colCols = colCols;
        }

        @Override
        public void backward(GradTensor gradOut) {
            // gradOut: [N, C_out, Hout, Wout]
            float[] dOut = gradOut.data();
            float[] wFlat = weightT.data(); // [C_out, colRows]

            // ── grad w.r.t. bias: sum over N, Hout, Wout ─────────────
            if (biasT != null && biasT.requiresGrad()) {
                float[] dBias = new float[outChannels];
                for (int n = 0; n < N; n++)
                    for (int c = 0; c < outChannels; c++)
                        for (int i = 0; i < Hout * Wout; i++)
                            dBias[c] += dOut[n * outChannels * Hout * Wout + c * Hout * Wout + i];
                biasT.backward(GradTensor.of(dBias, outChannels));
            }

            // ── grad w.r.t. weight: dOut_reshaped @ col^T ────────────
            // dOut reshaped: [C_out, N*Hout*Wout], col: [colRows, N*Hout*Wout]
            if (weightT.requiresGrad()) {
                float[] dW = new float[outChannels * colRows];
                for (int n = 0; n < N; n++) {
                    float[] colN = slice(col, n * colRows * colCols, colRows * colCols);
                    // dOut_n: [C_out, Hout*Wout]
                    float[] dOutN = slice(dOut, n * outChannels * Hout * Wout, outChannels * Hout * Wout);
                    // dW += dOut_n @ colN^T  → [C_out, colRows]
                    float[] colNT = transpose(colN, colRows, colCols);
                    float[] dWn = VectorOps.matmul(dOutN, colNT, outChannels, colCols, colRows);
                    for (int i = 0; i < dW.length; i++) dW[i] += dWn[i];
                }
                weightT.backward(GradTensor.of(dW, outChannels, colRows));
            }

            // ── grad w.r.t. input: col2im(weight^T @ dOut) ───────────
            if (input.requiresGrad()) {
                float[] dInput = new float[N * Cin * H * W];
                // weight^T: [colRows, C_out]
                float[] wT = transpose(wFlat, outChannels, colRows);
                for (int n = 0; n < N; n++) {
                    float[] dOutN = slice(dOut, n * outChannels * Hout * Wout, outChannels * Hout * Wout);
                    // dCol = wT @ dOut_n → [colRows, Hout*Wout]
                    float[] dCol = VectorOps.matmul(wT, dOutN, colRows, outChannels, Hout * Wout);
                    col2im(dCol, dInput, n, Cin, H, W, Hout, Wout);
                }
                input.backward(GradTensor.of(dInput, N, Cin, H, W));
            }
        }

        /** Scatter dCol back into dInput (inverse of im2col). */
        private void col2im(float[] dCol, float[] dInput, int n,
                             int Cin, int H, int W, int Hout, int Wout) {
            int row = 0;
            for (int c = 0; c < Cin; c++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++, row++) {
                        for (int oh = 0; oh < Hout; oh++) {
                            int ih = oh * strideH - padH + kh;
                            for (int ow = 0; ow < Wout; ow++) {
                                int iw = ow * strideW - padW + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    dInput[n * Cin * H * W + c * H * W + ih * W + iw]
                                        += dCol[row * Hout * Wout + oh * Wout + ow];
                                }
                            }
                        }
                    }
                }
            }
        }

        private static float[] transpose(float[] m, int rows, int cols) {
            float[] t = new float[rows * cols];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    t[c * rows + r] = m[r * cols + c];
            return t;
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private static float[] randomUniform(int n, float lo, float hi) {
        float[] d = new float[n];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < n; i++) d[i] = lo + rng.nextFloat() * (hi - lo);
        return d;
    }

    private static float[] slice(float[] src, int offset, int len) {
        float[] out = new float[len];
        System.arraycopy(src, offset, out, 0, len);
        return out;
    }

    @Override
    public String toString() {
        return String.format("Conv2d(%d, %d, kernel=(%d,%d), stride=(%d,%d), padding=(%d,%d))",
            inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW);
    }
}
