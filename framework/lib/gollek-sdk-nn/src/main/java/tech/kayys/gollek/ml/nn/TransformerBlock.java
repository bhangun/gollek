package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.TensorOps;

/**
 * Transformer Encoder block — equivalent to {@code torch.nn.TransformerEncoderLayer}.
 *
 * <p>Implements the standard pre-norm Transformer block used in BERT, RoBERTa, and ViT:
 * <pre>
 *   x = x + Dropout(MultiHeadAttention(LayerNorm(x)))
 *   x = x + Dropout(FFN(LayerNorm(x)))
 * </pre>
 *
 * <p>The attention scores are computed via
 * {@link TensorOps#einsum einsum("bhid,bhjd->bhij")} using the JDK 25 Vector API.
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li>Input:  {@code [B, T, dModel]}</li>
 *   <li>Output: {@code [B, T, dModel]}</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var block = new TransformerBlock(dModel=512, nHeads=8, dFF=2048, dropout=0.1f);
 * GradTensor out = block.forward(x);  // [B, T, 512] → [B, T, 512]
 * }</pre>
 */
public class TransformerBlock extends NNModule {

    private final int dModel;
    private final int nHeads;
    private final int headDim;

    // Self-attention projections
    private final Linear wQ, wK, wV, wO;

    // Feed-forward network
    private final Linear ff1, ff2;

    // Layer norms (pre-norm style)
    private final LayerNorm norm1, norm2;

    private final float dropoutRate;

    /**
     * Constructs a Transformer encoder block.
     *
     * @param dModel      model dimension (must be divisible by {@code nHeads})
     * @param nHeads      number of attention heads
     * @param dFF         feed-forward hidden dimension (typically 4 × dModel)
     * @param dropoutRate dropout probability applied after attention and FFN
     * @throws IllegalArgumentException if {@code dModel % nHeads != 0}
     */
    public TransformerBlock(int dModel, int nHeads, int dFF, float dropoutRate) {
        if (dModel % nHeads != 0)
            throw new IllegalArgumentException("dModel must be divisible by nHeads");
        this.dModel      = dModel;
        this.nHeads      = nHeads;
        this.headDim     = dModel / nHeads;
        this.dropoutRate = dropoutRate;

        this.wQ    = register("wQ",    new Linear(dModel, dModel, false));
        this.wK    = register("wK",    new Linear(dModel, dModel, false));
        this.wV    = register("wV",    new Linear(dModel, dModel, false));
        this.wO    = register("wO",    new Linear(dModel, dModel));
        this.ff1   = register("ff1",   new Linear(dModel, dFF));
        this.ff2   = register("ff2",   new Linear(dFF,   dModel));
        this.norm1 = register("norm1", new LayerNorm(dModel));
        this.norm2 = register("norm2", new LayerNorm(dModel));
    }

    /**
     * Forward pass through the Transformer block.
     *
     * @param x input tensor {@code [B, T, dModel]}
     * @return output tensor {@code [B, T, dModel]}
     */
    @Override
    public GradTensor forward(GradTensor x) {
        // Pre-norm self-attention with residual
        GradTensor attnOut = selfAttention(norm1.forward(x));
        x = x.add(dropout(attnOut));

        // Pre-norm FFN with residual
        GradTensor ffOut = ffn(norm2.forward(x));
        x = x.add(dropout(ffOut));

        return x;
    }

    // ── Multi-head self-attention ─────────────────────────────────────────

    /**
     * Scaled dot-product multi-head self-attention.
     *
     * @param x normalized input {@code [B, T, dModel]}
     * @return attention output {@code [B, T, dModel]}
     */
    private GradTensor selfAttention(GradTensor x) {
        long[] s = x.shape();
        int B = (int) s[0], T = (int) s[1];

        // Project to Q, K, V: [B, T, dModel]
        GradTensor Q = wQ.forward(x);
        GradTensor K = wK.forward(x);
        GradTensor V = wV.forward(x);

        // Reshape to [B, nHeads, T, headDim]
        Q = Q.reshape(B, T, nHeads, headDim).transpose(); // [B, nHeads, T, headDim]
        K = K.reshape(B, T, nHeads, headDim).transpose();
        V = V.reshape(B, T, nHeads, headDim).transpose();

        // Attention scores: [B, nHeads, T, T]
        float scale = (float) (1.0 / Math.sqrt(headDim));
        GradTensor scores = TensorOps.einsum("bhid,bhjd->bhij", Q, K).mul(scale);

        // Softmax over last dim
        GradTensor attnWeights = softmaxLastDim(scores);

        // Weighted sum of values: [B, nHeads, T, headDim]
        GradTensor attnOut = TensorOps.einsum("bhij,bhjd->bhid", attnWeights, V);

        // Reshape back: [B, T, dModel]
        attnOut = attnOut.reshape(B, T, dModel);
        return wO.forward(attnOut);
    }

    /** Feed-forward network: Linear → GELU → Linear. */
    private GradTensor ffn(GradTensor x) {
        return ff2.forward(ff1.forward(x).relu());
    }

    /** Applies dropout during training (identity during eval). */
    private GradTensor dropout(GradTensor x) {
        if (!isTraining() || dropoutRate == 0f) return x;
        float[] d = x.data().clone();
        float scale = 1f / (1f - dropoutRate);
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < d.length; i++) {
            if (rng.nextFloat() < dropoutRate) d[i] = 0f;
            else d[i] *= scale;
        }
        return GradTensor.of(d, x.shape());
    }

    /** Softmax over the last dimension. */
    private GradTensor softmaxLastDim(GradTensor x) {
        // Delegate to existing softmax (operates on last dim)
        return x.softmax();
    }

    @Override
    public String toString() {
        return String.format("TransformerBlock(dModel=%d, nHeads=%d, headDim=%d)",
            dModel, nHeads, headDim);
    }
}
