package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Gated Recurrent Unit (GRU) — equivalent to {@code torch.nn.GRU}.
 *
 * <p>Lighter than LSTM (no cell state), often comparable accuracy.
 *
 * <p>Gate equations:
 * <pre>
 *   r = σ(W_ir·x + b_ir + W_hr·h + b_hr)   reset gate
 *   z = σ(W_iz·x + b_iz + W_hz·h + b_hz)   update gate
 *   n = tanh(W_in·x + b_in + r ⊙ (W_hn·h + b_hn))  new gate
 *   h' = (1 - z) ⊙ n + z ⊙ h
 * </pre>
 *
 * <p>Input:  {@code [T, N, inputSize]}
 * <p>Output: {@code [T, N, hiddenSize]}, h_n {@code [1, N, hiddenSize]}
 */
public class GRU extends NNModule {

    private final int inputSize;
    private final int hiddenSize;
    private final boolean batchFirst;

    private final Parameter weightIH; // [3H, inputSize]
    private final Parameter weightHH; // [3H, hiddenSize]
    private final Parameter biasIH;   // [3H]
    private final Parameter biasHH;   // [3H]

    public GRU(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, false);
    }

    public GRU(int inputSize, int hiddenSize, boolean batchFirst) {
        this.inputSize  = inputSize;
        this.hiddenSize = hiddenSize;
        this.batchFirst = batchFirst;

        int H3 = 3 * hiddenSize;
        float bound = (float) (1.0 / Math.sqrt(hiddenSize));

        this.weightIH = registerParameter("weight_ih",
            GradTensor.of(randomUniform(H3 * inputSize,  -bound, bound), H3, inputSize));
        this.weightHH = registerParameter("weight_hh",
            GradTensor.of(randomUniform(H3 * hiddenSize, -bound, bound), H3, hiddenSize));
        this.biasIH   = registerParameter("bias_ih", GradTensor.of(new float[H3], H3));
        this.biasHH   = registerParameter("bias_hh", GradTensor.of(new float[H3], H3));
    }

    public record GRUOutput(GradTensor output, GradTensor hn) {}

    /**
     * Forward pass over a full sequence (required by NNModule contract).
     * Returns only the output sequence; use forwardFull() for hidden states.
     *
     * @param input [T, N, inputSize] (or [N, T, inputSize] if batchFirst)
     * @return output sequence [T, N, hiddenSize]
     */
    @Override
    public GradTensor forward(GradTensor input) {
        return forwardFull(input).output();
    }

    /**
     * Forward pass with full outputs including hidden state.
     *
     * @param input [T, N, inputSize] (or [N, T, inputSize] if batchFirst)
     * @return {@link GRUOutput} containing output sequence and h_n
     */
    public GRUOutput forwardFull(GradTensor input) {
        long[] s = input.shape();
        int T = batchFirst ? (int) s[1] : (int) s[0];
        int N = batchFirst ? (int) s[0] : (int) s[1];

        float[] wIH = weightIH.data().data();
        float[] wHH = weightHH.data().data();
        float[] bIH = biasIH.data().data();
        float[] bHH = biasHH.data().data();

        float[] h = new float[N * hiddenSize];
        float[] outData = new float[T * N * hiddenSize];

        for (int t = 0; t < T; t++) {
            float[] xt = extractTimestep(input.data(), t, T, N, inputSize, batchFirst);
            stepGRU(xt, h, wIH, wHH, bIH, bHH, N);
            System.arraycopy(h, 0, outData, t * N * hiddenSize, N * hiddenSize);
        }

        return new GRUOutput(
            GradTensor.of(outData, T, N, hiddenSize),
            GradTensor.of(h.clone(), 1, N, hiddenSize)
        );
    }

    private void stepGRU(float[] xt, float[] h,
                          float[] wIH, float[] wHH,
                          float[] bIH, float[] bHH, int N) {
        int H = hiddenSize, H3 = 3 * H;
        // gatesX = W_ih @ xt  [N, 3H]
        float[] gX = VectorOps.matmul(xt, wIH, N, inputSize,  H3);
        // gatesH = W_hh @ h   [N, 3H]
        float[] gH = VectorOps.matmul(h,  wHH, N, hiddenSize, H3);

        for (int n = 0; n < N; n++) {
            int base = n * H3, hn = n * H;
            // r, z gates use full gH
            float[] r = new float[H], z = new float[H];
            for (int j = 0; j < H; j++) {
                r[j] = sigmoid(gX[base + j]     + bIH[j]     + gH[base + j]     + bHH[j]);
                z[j] = sigmoid(gX[base + H + j] + bIH[H + j] + gH[base + H + j] + bHH[H + j]);
            }
            // n gate: r ⊙ (W_hn·h + b_hn)
            for (int j = 0; j < H; j++) {
                float n_gate = (float) Math.tanh(
                    gX[base + 2*H + j] + bIH[2*H + j] +
                    r[j] * (gH[base + 2*H + j] + bHH[2*H + j])
                );
                h[hn + j] = (1f - z[j]) * n_gate + z[j] * h[hn + j];
            }
        }
    }

    private static float sigmoid(float x) { return 1f / (1f + (float) Math.exp(-x)); }

    private static float[] extractTimestep(float[] data, int t, int T, int N,
                                            int inputSize, boolean batchFirst) {
        float[] xt = new float[N * inputSize];
        for (int n = 0; n < N; n++) {
            int src = batchFirst ? n * T * inputSize + t * inputSize
                                 : t * N * inputSize + n * inputSize;
            System.arraycopy(data, src, xt, n * inputSize, inputSize);
        }
        return xt;
    }

    private static float[] randomUniform(int n, float lo, float hi) {
        float[] d = new float[n];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < n; i++) d[i] = lo + rng.nextFloat() * (hi - lo);
        return d;
    }

    @Override
    public String toString() {
        return String.format("GRU(input=%d, hidden=%d, batchFirst=%b)",
            inputSize, hiddenSize, batchFirst);
    }
}
