package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Group Normalization — normalizes over groups of channels, independent of batch size.
 *
 * <p>Equivalent to {@code torch.nn.GroupNorm}. Unlike BatchNorm, GroupNorm works
 * well with small batch sizes (even batch=1), making it preferred for detection,
 * segmentation, and video models.
 *
 * <p>Formula:
 * <pre>
 *   Split C channels into G groups of C/G channels each
 *   For each group: normalize over [C/G, H, W] dimensions
 *   y = γ·x̂ + β
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var gn = new GroupNorm(numGroups = 32, numChannels = 256);
 * GradTensor out = gn.forward(x); // [N, 256, H, W] → [N, 256, H, W]
 * }</pre>
 */
public class GroupNorm extends NNModule {

    private final int numGroups;
    private final int numChannels;
    private final float eps;
    private final Parameter gamma; // [C]
    private final Parameter beta;  // [C]

    /**
     * Creates a GroupNorm layer.
     *
     * @param numGroups   number of groups to divide channels into (must divide numChannels)
     * @param numChannels total number of channels C
     * @throws IllegalArgumentException if numChannels is not divisible by numGroups
     */
    public GroupNorm(int numGroups, int numChannels) {
        this(numGroups, numChannels, 1e-5f);
    }

    /**
     * Creates a GroupNorm layer with custom epsilon.
     *
     * @param numGroups   number of groups
     * @param numChannels total channels
     * @param eps         numerical stability constant
     */
    public GroupNorm(int numGroups, int numChannels, float eps) {
        if (numChannels % numGroups != 0)
            throw new IllegalArgumentException("numChannels must be divisible by numGroups");
        this.numGroups   = numGroups;
        this.numChannels = numChannels;
        this.eps         = eps;
        float[] ones = new float[numChannels]; java.util.Arrays.fill(ones, 1f);
        this.gamma = registerParameter("weight", GradTensor.of(ones, numChannels));
        this.beta  = registerParameter("bias",   GradTensor.of(new float[numChannels], numChannels));
    }

    /**
     * Forward pass.
     *
     * @param x input tensor {@code [N, C, H, W]}
     * @return normalized tensor {@code [N, C, H, W]}
     */
    @Override
    public GradTensor forward(GradTensor x) {
        long[] s = x.shape();
        int N = (int)s[0], C = (int)s[1], H = (int)s[2], W = (int)s[3];
        int G = numGroups, cPerG = C / G, groupSize = cPerG * H * W;
        float[] d = x.data(), g = gamma.data().data(), b = beta.data().data();
        float[] out = new float[d.length];

        for (int n = 0; n < N; n++) {
            for (int gr = 0; gr < G; gr++) {
                // Collect group values
                float[] vals = new float[groupSize];
                int vi = 0;
                for (int c = gr * cPerG; c < (gr + 1) * cPerG; c++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                            vals[vi++] = d[n*C*H*W + c*H*W + h*W + w];

                float mean = VectorOps.sum(vals) / groupSize;
                float[] sq = new float[groupSize];
                for (int i = 0; i < groupSize; i++) sq[i] = (vals[i] - mean) * (vals[i] - mean);
                float var = VectorOps.sum(sq) / groupSize;
                float invStd = 1f / (float) Math.sqrt(var + eps);

                vi = 0;
                for (int c = gr * cPerG; c < (gr + 1) * cPerG; c++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++) {
                            int idx = n*C*H*W + c*H*W + h*W + w;
                            out[idx] = g[c] * (d[idx] - mean) * invStd + b[c];
                            vi++;
                        }
            }
        }
        return GradTensor.of(out, N, C, H, W);
    }

    @Override public String toString() {
        return "GroupNorm(groups=" + numGroups + ", channels=" + numChannels + ")";
    }
}
