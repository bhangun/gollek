package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Rotary Position Embedding (RoPE) — encodes position information by rotating
 * query and key vectors in the complex plane.
 *
 * <p>Based on <em>"RoFormer: Enhanced Transformer with Rotary Position Embedding"</em>
 * (Su et al., 2021). Used in LLaMA, GPT-NeoX, Falcon, and most modern LLMs.
 *
 * <p>Advantages over learned positional embeddings:
 * <ul>
 *   <li>Relative position awareness (attention decays with distance)</li>
 *   <li>Extrapolates to longer sequences than seen during training</li>
 *   <li>No additional parameters</li>
 * </ul>
 *
 * <p>Rotation formula for each pair of dimensions {@code (2i, 2i+1)}:
 * <pre>
 *   [q_{2i}  ]   [cos(mθᵢ)  -sin(mθᵢ)] [q_{2i}  ]
 *   [q_{2i+1}] = [sin(mθᵢ)   cos(mθᵢ)] [q_{2i+1}]
 *   where θᵢ = 1 / 10000^(2i/d)
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var rope = new RotaryEmbedding(headDim = 64, maxSeqLen = 2048);
 * GradTensor q = rope.apply(queries); // [B, H, T, headDim]
 * GradTensor k = rope.apply(keys);
 * }</pre>
 */
public final class RotaryEmbedding extends NNModule {

    private final int   dim;
    private final float base;
    private final float[] cosCache; // [maxSeqLen, dim/2]
    private final float[] sinCache; // [maxSeqLen, dim/2]
    private final int   maxSeqLen;

    /**
     * Creates a RoPE module with default base of 10000.
     *
     * @param dim       head dimension (must be even)
     * @param maxSeqLen maximum sequence length to precompute
     */
    public RotaryEmbedding(int dim, int maxSeqLen) {
        this(dim, maxSeqLen, 10000f);
    }

    /**
     * Creates a RoPE module with custom base (use 500000 for LLaMA-3 style).
     *
     * @param dim       head dimension (must be even)
     * @param maxSeqLen maximum sequence length
     * @param base      frequency base (default 10000)
     */
    public RotaryEmbedding(int dim, int maxSeqLen, float base) {
        if (dim % 2 != 0) throw new IllegalArgumentException("dim must be even");
        this.dim       = dim;
        this.maxSeqLen = maxSeqLen;
        this.base      = base;
        int half = dim / 2;
        this.cosCache = new float[maxSeqLen * half];
        this.sinCache = new float[maxSeqLen * half];
        precompute(half);
    }

    /** Precomputes cos/sin tables for all positions and frequency pairs. */
    private void precompute(int half) {
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < half; i++) {
                float theta = pos / (float) Math.pow(base, 2.0 * i / dim);
                cosCache[pos * half + i] = (float) Math.cos(theta);
                sinCache[pos * half + i] = (float) Math.sin(theta);
            }
        }
    }

    /**
     * Applies rotary embeddings to a query or key tensor.
     *
     * @param x input tensor {@code [B, H, T, headDim]}
     * @return rotated tensor {@code [B, H, T, headDim]}
     */
    public GradTensor apply(GradTensor x) {
        long[] s = x.shape();
        int B = (int)s[0], H = (int)s[1], T = (int)s[2], D = (int)s[3];
        int half = D / 2;
        float[] d = x.data(), out = new float[d.length];

        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++)
                for (int t = 0; t < T; t++) {
                    int base = b*H*T*D + h*T*D + t*D;
                    for (int i = 0; i < half; i++) {
                        float cos = cosCache[t * half + i];
                        float sin = sinCache[t * half + i];
                        float x0  = d[base + 2*i];
                        float x1  = d[base + 2*i + 1];
                        out[base + 2*i]     = x0 * cos - x1 * sin;
                        out[base + 2*i + 1] = x0 * sin + x1 * cos;
                    }
                }
        return GradTensor.of(out, s);
    }

    @Override
    public GradTensor forward(GradTensor x) { return apply(x); }

    @Override public String toString() {
        return "RotaryEmbedding(dim=" + dim + ", maxSeqLen=" + maxSeqLen + ", base=" + base + ")";
    }
}
