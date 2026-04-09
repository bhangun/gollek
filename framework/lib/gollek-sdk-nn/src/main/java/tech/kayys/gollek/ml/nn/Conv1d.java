package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * 1D Convolution — equivalent to {@code torch.nn.Conv1d}.
 * Used for audio, NLP sequence features, time-series.
 *
 * <p>Input:  {@code [N, C_in,  L]}
 * <p>Output: {@code [N, C_out, L_out]}  where {@code L_out = (L + 2*pad - kernel) / stride + 1}
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var conv = new Conv1d(64, 128, 3);          // kernel=3, no padding
 * var conv = new Conv1d(64, 128, 3, 1, 1);    // same-length output
 * }</pre>
 */
public class Conv1d extends NNModule {

    private final int inChannels, outChannels, kernelSize, stride, padding;
    private final Parameter weight; // [C_out, C_in, K]
    private final Parameter bias;   // [C_out]

    public Conv1d(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0, true);
    }

    public Conv1d(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this(inChannels, outChannels, kernelSize, stride, padding, true);
    }

    public Conv1d(int inChannels, int outChannels, int kernelSize,
                  int stride, int padding, boolean useBias) {
        this.inChannels  = inChannels;
        this.outChannels = outChannels;
        this.kernelSize  = kernelSize;
        this.stride      = stride;
        this.padding     = padding;

        float bound = (float) Math.sqrt(2.0 / (inChannels * kernelSize));
        this.weight = registerParameter("weight",
            GradTensor.of(randomUniform(outChannels * inChannels * kernelSize, -bound, bound),
                          outChannels, inChannels, kernelSize));
        this.bias = useBias
            ? registerParameter("bias", GradTensor.of(new float[outChannels], outChannels))
            : null;
    }

    @Override
    public GradTensor forward(GradTensor input) {
        long[] s = input.shape();
        int N   = (int) s[0], Cin = (int) s[1], L = (int) s[2];
        int Lout = (L + 2 * padding - kernelSize) / stride + 1;

        // im2col for 1D: [N, Cin*K, Lout]
        int colRows = Cin * kernelSize;
        float[] col = im2col1d(input.data(), N, Cin, L, Lout, colRows);
        float[] wFlat = weight.data().data(); // [C_out, Cin*K]

        float[] outData = new float[N * outChannels * Lout];
        for (int n = 0; n < N; n++) {
            float[] colN = slice(col, n * colRows * Lout, colRows * Lout);
            float[] gemm = VectorOps.matmul(wFlat, colN, outChannels, colRows, Lout);
            System.arraycopy(gemm, 0, outData, n * outChannels * Lout, gemm.length);
        }

        if (bias != null) {
            float[] bData = bias.data().data();
            for (int n = 0; n < N; n++)
                for (int c = 0; c < outChannels; c++) {
                    int base = n * outChannels * Lout + c * Lout;
                    for (int i = base; i < base + Lout; i++) outData[i] += bData[c];
                }
        }

        GradTensor out = GradTensor.of(outData, N, outChannels, Lout);
        if (input.requiresGrad() || weight.data().requiresGrad()) {
            final float[] savedCol = col;
            final int fN = N, fCin = Cin, fL = L, fLout = Lout, fColRows = colRows;
            out.requiresGrad(true);
            out.setGradFn(new tech.kayys.gollek.ml.autograd.Function.Context("Conv1dBackward") {
                @Override public void backward(GradTensor g) {
                    float[] dOut = g.data();
                    if (weight.data().requiresGrad()) {
                        float[] dW = new float[outChannels * fColRows];
                        for (int n = 0; n < fN; n++) {
                            float[] colN = slice(savedCol, n * fColRows * fLout, fColRows * fLout);
                            float[] dOutN = slice(dOut, n * outChannels * fLout, outChannels * fLout);
                            float[] colNT = transpose(colN, fColRows, fLout);
                            float[] dWn = VectorOps.matmul(dOutN, colNT, outChannels, fLout, fColRows);
                            for (int i = 0; i < dW.length; i++) dW[i] += dWn[i];
                        }
                        weight.data().backward(GradTensor.of(dW, outChannels, fColRows));
                    }
                    if (input.requiresGrad()) {
                        float[] dInput = new float[fN * fCin * fL];
                        float[] wT = transpose(wFlat, outChannels, fColRows);
                        for (int n = 0; n < fN; n++) {
                            float[] dOutN = slice(dOut, n * outChannels * fLout, outChannels * fLout);
                            float[] dCol = VectorOps.matmul(wT, dOutN, fColRows, outChannels, fLout);
                            col2im1d(dCol, dInput, n, fCin, fL, fLout);
                        }
                        input.backward(GradTensor.of(dInput, fN, fCin, fL));
                    }
                }
            });
        }
        return out;
    }

    private float[] im2col1d(float[] data, int N, int Cin, int L, int Lout, int colRows) {
        float[] col = new float[N * colRows * Lout];
        for (int n = 0; n < N; n++) {
            int colBase = n * colRows * Lout;
            int row = 0;
            for (int c = 0; c < Cin; c++) {
                for (int k = 0; k < kernelSize; k++, row++) {
                    for (int ol = 0; ol < Lout; ol++) {
                        int il = ol * stride - padding + k;
                        col[colBase + row * Lout + ol] =
                            (il >= 0 && il < L) ? data[n * Cin * L + c * L + il] : 0f;
                    }
                }
            }
        }
        return col;
    }

    private void col2im1d(float[] dCol, float[] dInput, int n, int Cin, int L, int Lout) {
        int row = 0;
        for (int c = 0; c < Cin; c++) {
            for (int k = 0; k < kernelSize; k++, row++) {
                for (int ol = 0; ol < Lout; ol++) {
                    int il = ol * stride - padding + k;
                    if (il >= 0 && il < L)
                        dInput[n * Cin * L + c * L + il] += dCol[row * Lout + ol];
                }
            }
        }
    }

    private static float[] slice(float[] src, int off, int len) {
        float[] out = new float[len]; System.arraycopy(src, off, out, 0, len); return out;
    }

    private static float[] transpose(float[] m, int rows, int cols) {
        float[] t = new float[rows * cols];
        for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) t[c * rows + r] = m[r * cols + c];
        return t;
    }

    private static float[] randomUniform(int n, float lo, float hi) {
        float[] d = new float[n]; java.util.Random rng = new java.util.Random();
        for (int i = 0; i < n; i++) d[i] = lo + rng.nextFloat() * (hi - lo);
        return d;
    }

    @Override public String toString() {
        return String.format("Conv1d(%d, %d, kernel=%d, stride=%d, padding=%d)",
            inChannels, outChannels, kernelSize, stride, padding);
    }
}
