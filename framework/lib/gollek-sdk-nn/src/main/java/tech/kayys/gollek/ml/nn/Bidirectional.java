package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Bidirectional RNN wrapper — runs a recurrent layer in both forward and
 * backward directions and concatenates the outputs.
 *
 * <p>Equivalent to {@code torch.nn.RNN/LSTM/GRU(bidirectional=True)}.
 * Output hidden size = 2 × hiddenSize.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var biLSTM = new Bidirectional(new LSTM(128, 256));
 * GradTensor out = biLSTM.forward(x); // [T, N, 512]
 * }</pre>
 */
public final class Bidirectional extends NNModule {

    private final NNModule forward;
    private final NNModule backward;

    /**
     * Wraps a recurrent module with bidirectional processing.
     * Creates two independent copies: one for each direction.
     *
     * @param rnn the recurrent module to wrap (LSTM or GRU)
     */
    public Bidirectional(NNModule rnn) {
        this.forward  = register("forward",  rnn);
        // Backward direction uses a separate module instance
        this.backward = register("backward", cloneModule(rnn));
    }

    /**
     * Forward pass: runs both directions and concatenates along the feature dim.
     *
     * @param x input {@code [T, N, inputSize]}
     * @return concatenated output {@code [T, N, 2*hiddenSize]}
     */
    @Override
    public GradTensor forward(GradTensor x) {
        // Forward direction
        GradTensor fwdOut = extractOutput(forward.forward(x));

        // Backward direction: reverse time axis
        GradTensor xRev = reverseTime(x);
        GradTensor bwdOut = reverseTime(extractOutput(backward.forward(xRev)));

        // Concatenate along feature dim: [T, N, 2*H]
        return tech.kayys.gollek.ml.autograd.TensorOps.cat(
            java.util.List.of(fwdOut, bwdOut), 2);
    }

    /** Extracts output tensor from LSTM/GRU output (handles both record types). */
    private GradTensor extractOutput(GradTensor out) { return out; }

    /** Reverses the time dimension (dim 0) of a [T, N, D] tensor. */
    private static GradTensor reverseTime(GradTensor x) {
        long[] s = x.shape();
        int T = (int)s[0], N = (int)s[1], D = (int)s[2];
        float[] src = x.data(), dst = new float[src.length];
        for (int t = 0; t < T; t++)
            System.arraycopy(src, t*N*D, dst, (T-1-t)*N*D, N*D);
        return GradTensor.of(dst, s);
    }

    /** Creates a new module of the same type with fresh parameters. */
    private static NNModule cloneModule(NNModule m) {
        if (m instanceof LSTM lstm) {
            long[] ws = lstm.parameters().get(0).data().shape();
            int H4 = (int)ws[0], inputSize = (int)ws[1];
            return new LSTM(inputSize, H4 / 4);
        }
        if (m instanceof GRU gru) {
            long[] ws = gru.parameters().get(0).data().shape();
            int H3 = (int)ws[0], inputSize = (int)ws[1];
            return new GRU(inputSize, H3 / 3);
        }
        throw new IllegalArgumentException("Bidirectional only supports LSTM and GRU");
    }

    @Override public String toString() {
        return "Bidirectional(" + forward + ")";
    }
}
