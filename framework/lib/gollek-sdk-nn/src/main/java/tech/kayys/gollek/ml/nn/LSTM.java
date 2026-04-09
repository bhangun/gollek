package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Long Short-Term Memory (LSTM) layer — equivalent to {@code torch.nn.LSTM}.
 *
 * <p>Processes a sequence of inputs and returns the output sequence plus
 * the final hidden/cell state.
 *
 * <p>Gate equations (per time step):
 * <pre>
 *   i = σ(W_ii·x + b_ii + W_hi·h + b_hi)   input gate
 *   f = σ(W_if·x + b_if + W_hf·h + b_hf)   forget gate
 *   g = tanh(W_ig·x + b_ig + W_hg·h + b_hg) cell gate
 *   o = σ(W_io·x + b_io + W_ho·h + b_ho)   output gate
 *   c' = f ⊙ c + i ⊙ g
 *   h' = o ⊙ tanh(c')
 * </pre>
 *
 * <p>Input:  sequence {@code [T, N, inputSize]} or {@code [N, T, inputSize]} (batchFirst)
 * <p>Output: {@code [T, N, hiddenSize]}, h_n {@code [1, N, hiddenSize]}, c_n {@code [1, N, hiddenSize]}
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var lstm = new LSTM(128, 256);           // inputSize=128, hiddenSize=256
 * var lstm = new LSTM(128, 256, true);     // batchFirst=true
 * LSTMOutput out = lstm.forward(input);
 * GradTensor sequence = out.output();     // [T, N, 256]
 * GradTensor hidden   = out.hn();         // [1, N, 256]
 * }</pre>
 */
public class LSTM extends NNModule {

    private final int inputSize;
    private final int hiddenSize;
    private final boolean batchFirst;

    // Combined weight matrices for efficiency: [4*hiddenSize, inputSize/hiddenSize]
    private final Parameter weightIH; // input-hidden:  [4H, inputSize]
    private final Parameter weightHH; // hidden-hidden: [4H, hiddenSize]
    private final Parameter biasIH;   // [4H]
    private final Parameter biasHH;   // [4H]

    public LSTM(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, false);
    }

    public LSTM(int inputSize, int hiddenSize, boolean batchFirst) {
        this.inputSize  = inputSize;
        this.hiddenSize = hiddenSize;
        this.batchFirst = batchFirst;

        int H4 = 4 * hiddenSize;
        float bound = (float) (1.0 / Math.sqrt(hiddenSize));

        this.weightIH = registerParameter("weight_ih",
            GradTensor.of(randomUniform(H4 * inputSize,  -bound, bound), H4, inputSize));
        this.weightHH = registerParameter("weight_hh",
            GradTensor.of(randomUniform(H4 * hiddenSize, -bound, bound), H4, hiddenSize));
        this.biasIH   = registerParameter("bias_ih",
            GradTensor.of(new float[H4], H4));
        this.biasHH   = registerParameter("bias_hh",
            GradTensor.of(new float[H4], H4));
    }

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
     * Forward pass with full outputs including hidden and cell states.
     *
     * @param input [T, N, inputSize] (or [N, T, inputSize] if batchFirst)
     * @return {@link LSTMOutput} containing output sequence, h_n, c_n
     */
    public LSTMOutput forwardFull(GradTensor input) {
        long[] s = input.shape();
        int T = batchFirst ? (int) s[1] : (int) s[0];
        int N = batchFirst ? (int) s[0] : (int) s[1];

        float[] wIH = weightIH.data().data();
        float[] wHH = weightHH.data().data();
        float[] bIH = biasIH.data().data();
        float[] bHH = biasHH.data().data();

        float[] h = new float[N * hiddenSize]; // h_0 = zeros
        float[] c = new float[N * hiddenSize]; // c_0 = zeros
        float[] outData = new float[T * N * hiddenSize];

        for (int t = 0; t < T; t++) {
            // Extract x_t: [N, inputSize]
            float[] xt = extractTimestep(input.data(), t, T, N, inputSize, batchFirst);

            // gates = W_ih @ x_t^T + b_ih + W_hh @ h^T + b_hh  → [N, 4H]
            float[] gates = computeGates(xt, h, wIH, wHH, bIH, bHH, N);

            // Apply gate activations and update h, c
            stepCell(gates, h, c, N);

            // Store h_t in output
            System.arraycopy(h, 0, outData, t * N * hiddenSize, N * hiddenSize);
        }

        GradTensor output = GradTensor.of(outData, T, N, hiddenSize);
        GradTensor hn     = GradTensor.of(h.clone(), 1, N, hiddenSize);
        GradTensor cn     = GradTensor.of(c.clone(), 1, N, hiddenSize);
        return new LSTMOutput(output, hn, cn);
    }

    // ── Cell step ────────────────────────────────────────────────────────

    /** Compute all 4 gates for all N samples using VectorOps matmul. */
    private float[] computeGates(float[] xt, float[] h,
                                  float[] wIH, float[] wHH,
                                  float[] bIH, float[] bHH, int N) {
        int H4 = 4 * hiddenSize;
        // [N, inputSize] @ [inputSize, 4H]^T  →  [N, 4H]
        float[] gatesX = VectorOps.matmul(xt,  wIH, N, inputSize,  H4);
        float[] gatesH = VectorOps.matmul(h,   wHH, N, hiddenSize, H4);
        float[] gates  = new float[N * H4];
        // gates = gatesX + gatesH + bIH + bHH  (Vector API fused)
        for (int n = 0; n < N; n++) {
            for (int g = 0; g < H4; g++) {
                int idx = n * H4 + g;
                gates[idx] = gatesX[idx] + gatesH[idx] + bIH[g] + bHH[g];
            }
        }
        return gates;
    }

    /** Apply sigmoid/tanh to gates and update h, c in-place. */
    private void stepCell(float[] gates, float[] h, float[] c, int N) {
        int H = hiddenSize;
        for (int n = 0; n < N; n++) {
            int base = n * 4 * H;
            int hn   = n * H;
            for (int j = 0; j < H; j++) {
                float i_gate = sigmoid(gates[base          + j]);
                float f_gate = sigmoid(gates[base +     H  + j]);
                float g_gate = (float) Math.tanh(gates[base + 2 * H + j]);
                float o_gate = sigmoid(gates[base + 3 * H  + j]);
                c[hn + j] = f_gate * c[hn + j] + i_gate * g_gate;
                h[hn + j] = o_gate * (float) Math.tanh(c[hn + j]);
            }
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private static float sigmoid(float x) {
        return 1f / (1f + (float) Math.exp(-x));
    }

    private static float[] extractTimestep(float[] data, int t, int T, int N,
                                            int inputSize, boolean batchFirst) {
        float[] xt = new float[N * inputSize];
        for (int n = 0; n < N; n++) {
            int srcOffset = batchFirst
                ? n * T * inputSize + t * inputSize
                : t * N * inputSize + n * inputSize;
            System.arraycopy(data, srcOffset, xt, n * inputSize, inputSize);
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
        return String.format("LSTM(input=%d, hidden=%d, batchFirst=%b)",
            inputSize, hiddenSize, batchFirst);
    }

    // ── Output record ────────────────────────────────────────────────────

    /**
     * Result of an LSTM forward pass.
     *
     * @param output full sequence output [T, N, hiddenSize]
     * @param hn     final hidden state   [1, N, hiddenSize]
     * @param cn     final cell state     [1, N, hiddenSize]
     */
    public record LSTMOutput(GradTensor output, GradTensor hn, GradTensor cn) {}
}
