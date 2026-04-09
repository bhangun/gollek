package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Connectionist Temporal Classification (CTC) Loss — enables training sequence
 * models without requiring aligned input-output pairs.
 *
 * <p>Based on <em>"Connectionist Temporal Classification: Labelling Unsegmented
 * Sequence Data with Recurrent Neural Networks"</em> (Graves et al., 2006).
 *
 * <p>Used for: speech recognition, OCR, handwriting recognition.
 *
 * <p>This implementation uses the forward-backward algorithm with log-space
 * computation for numerical stability.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var loss = new CTCLoss();
 * // logProbs: [T, N, C] log-softmax outputs
 * // targets:  [N, S] target sequences (class indices)
 * // inputLengths:  [N] actual input lengths
 * // targetLengths: [N] actual target lengths
 * GradTensor l = loss.forward(logProbs, targets, inputLengths, targetLengths);
 * }</pre>
 */
public final class CTCLoss {

    private static final int BLANK = 0; // blank label index (default 0)
    private static final float NEG_INF = Float.NEGATIVE_INFINITY;

    private final int blank;

    /** Creates a CTC loss with blank label index 0. */
    public CTCLoss() { this(0); }

    /**
     * Creates a CTC loss with a custom blank label index.
     *
     * @param blank index of the blank label in the vocabulary
     */
    public CTCLoss(int blank) { this.blank = blank; }

    /**
     * Computes the CTC loss for a batch of sequences.
     *
     * @param logProbs     log-softmax outputs {@code [T, N, C]}
     * @param targets      target sequences {@code [N, S]} (class indices as float)
     * @param inputLens    actual input lengths per sample {@code [N]}
     * @param targetLens   actual target lengths per sample {@code [N]}
     * @return scalar mean CTC loss (negative log-likelihood)
     */
    public GradTensor forward(GradTensor logProbs, GradTensor targets,
                               int[] inputLens, int[] targetLens) {
        long[] s = logProbs.shape();
        int T = (int) s[0], N = (int) s[1], C = (int) s[2];
        float[] lp = logProbs.data(), tg = targets.data();
        float[] losses = new float[N];

        for (int n = 0; n < N; n++) {
            int tLen = targetLens[n];
            int iLen = inputLens[n];
            int[] target = new int[tLen];
            for (int i = 0; i < tLen; i++) target[i] = (int) tg[n * (int) targets.shape()[1] + i];

            losses[n] = ctcForward(lp, n, N, C, iLen, target, tLen);
        }
        return GradTensor.scalar(VectorOps.sum(losses) / N);
    }

    /**
     * CTC forward algorithm for a single sample using log-space computation.
     * Returns the negative log-likelihood.
     */
    private float ctcForward(float[] lp, int n, int N, int C, int T,
                              int[] target, int S) {
        // Extended target with blanks: b t1 b t2 b ... tS b
        int L = 2 * S + 1;
        int[] ext = new int[L];
        for (int i = 0; i < L; i++) ext[i] = (i % 2 == 0) ? blank : target[i / 2];

        // Alpha table [T, L] in log space
        float[] alpha = new float[T * L]; java.util.Arrays.fill(alpha, NEG_INF);

        // Initialize
        alpha[0 * L + 0] = lp[0 * N * C + n * C + blank];
        if (L > 1) alpha[0 * L + 1] = lp[0 * N * C + n * C + ext[1]];

        // Forward pass
        for (int t = 1; t < T; t++) {
            for (int s = 0; s < L; s++) {
                float logP = lp[t * N * C + n * C + ext[s]];
                float prev = alpha[(t-1) * L + s];
                if (s > 0) prev = logSumExp(prev, alpha[(t-1) * L + s - 1]);
                if (s > 1 && ext[s] != blank && ext[s] != ext[s-2])
                    prev = logSumExp(prev, alpha[(t-1) * L + s - 2]);
                alpha[t * L + s] = prev + logP;
            }
        }

        // Total log-prob = log(alpha[T-1][L-1] + alpha[T-1][L-2])
        float logProb = alpha[(T-1) * L + L - 1];
        if (L >= 2) logProb = logSumExp(logProb, alpha[(T-1) * L + L - 2]);
        return -logProb;
    }

    private static float logSumExp(float a, float b) {
        if (a == NEG_INF) return b;
        if (b == NEG_INF) return a;
        float max = Math.max(a, b);
        return max + (float) Math.log(Math.exp(a - max) + Math.exp(b - max));
    }
}
