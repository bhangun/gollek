package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * 2D Batch Normalization — normalizes over the spatial dimensions (H, W) per channel.
 *
 * <p>Equivalent to {@code torch.nn.BatchNorm2d}. Used after Conv2d layers.
 *
 * <p>During training:
 * <pre>
 *   μ = mean(x, dims=[N,H,W])   per channel
 *   σ² = var(x, dims=[N,H,W])   per channel
 *   x̂ = (x - μ) / √(σ² + ε)
 *   y = γ·x̂ + β
 * </pre>
 *
 * <p>During inference, uses running statistics accumulated during training.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var bn = new BatchNorm2d(64);
 * GradTensor out = bn.forward(x); // [N, 64, H, W] → [N, 64, H, W]
 * }</pre>
 */
public class BatchNorm2d extends NNModule {

    private final int numFeatures;
    private final float eps;
    private final float momentum;

    private final Parameter gamma; // scale  [C]
    private final Parameter beta;  // shift  [C]

    // Running statistics (not trained, updated via EMA)
    private final float[] runningMean;
    private final float[] runningVar;

    /**
     * Creates a BatchNorm2d layer.
     *
     * @param numFeatures number of channels C
     */
    public BatchNorm2d(int numFeatures) {
        this(numFeatures, 1e-5f, 0.1f);
    }

    /**
     * Creates a BatchNorm2d layer with custom epsilon and momentum.
     *
     * @param numFeatures number of channels C
     * @param eps         numerical stability constant (default 1e-5)
     * @param momentum    EMA momentum for running stats (default 0.1)
     */
    public BatchNorm2d(int numFeatures, float eps, float momentum) {
        this.numFeatures = numFeatures;
        this.eps         = eps;
        this.momentum    = momentum;

        float[] ones  = new float[numFeatures]; java.util.Arrays.fill(ones, 1f);
        this.gamma = registerParameter("weight", GradTensor.of(ones.clone(), numFeatures));
        this.beta  = registerParameter("bias",   GradTensor.of(new float[numFeatures], numFeatures));
        this.runningMean = new float[numFeatures];
        this.runningVar  = new float[numFeatures]; java.util.Arrays.fill(runningVar, 1f);
    }

    /**
     * Forward pass — normalizes input over N, H, W dimensions per channel.
     *
     * @param x input tensor {@code [N, C, H, W]}
     * @return normalized tensor {@code [N, C, H, W]}
     */
    @Override
    public GradTensor forward(GradTensor x) {
        long[] s = x.shape();
        int N = (int)s[0], C = (int)s[1], H = (int)s[2], W = (int)s[3];
        int spatial = N * H * W;
        float[] d = x.data(), g = gamma.data().data(), b = beta.data().data();
        float[] out = new float[d.length];

        for (int c = 0; c < C; c++) {
            // Gather all values for this channel
            float[] vals = new float[spatial];
            int vi = 0;
            for (int n = 0; n < N; n++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        vals[vi++] = d[n*C*H*W + c*H*W + h*W + w];

            float mean, var;
            if (isTraining()) {
                mean = VectorOps.sum(vals) / spatial;
                float[] centered = new float[spatial];
                for (int i = 0; i < spatial; i++) centered[i] = vals[i] - mean;
                float[] sq = new float[spatial]; VectorOps.mul(centered, centered, sq);
                var = VectorOps.sum(sq) / spatial;
                // Update running stats
                runningMean[c] = (1 - momentum) * runningMean[c] + momentum * mean;
                runningVar[c]  = (1 - momentum) * runningVar[c]  + momentum * var;
            } else {
                mean = runningMean[c];
                var  = runningVar[c];
            }

            float invStd = 1f / (float) Math.sqrt(var + eps);
            vi = 0;
            for (int n = 0; n < N; n++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++) {
                        int idx = n*C*H*W + c*H*W + h*W + w;
                        out[idx] = g[c] * (d[idx] - mean) * invStd + b[c];
                        vi++;
                    }
        }
        return GradTensor.of(out, N, C, H, W);
    }

    @Override public String toString() {
        return "BatchNorm2d(" + numFeatures + ", eps=" + eps + ", momentum=" + momentum + ")";
    }
}
