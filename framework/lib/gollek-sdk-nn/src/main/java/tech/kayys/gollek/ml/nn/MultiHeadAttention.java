package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Multi-Head Attention mechanism.
 * <p>
 * Implements scaled dot-product attention with multiple heads:
 * {@code Attention(Q, K, V) = softmax(QK^T / √d_k) V}
 * <p>
 * Supports self-attention, cross-attention, and optional causal masking for
 * autoregressive generation (preventing the model from attending to future tokens).
 * <p>
 * Equivalent to {@code torch.nn.MultiheadAttention}.
 *
 * <h3>Example: Self-Attention</h3>
 * <pre>{@code
 * var attn = new MultiHeadAttention(768, 12);  // dim=768, heads=12
 * var output = attn.forward(input);  // [batch, seq, 768]
 * }</pre>
 *
 * <h3>Example: Causal Self-Attention (for autoregressive models)</h3>
 * <pre>{@code
 * var output = attn.forward(input, input, input, true);  // with causal mask
 * }</pre>
 *
 * <h3>Example: Cross-Attention (decoder attending to encoder)</h3>
 * <pre>{@code
 * var output = attn.forward(query, key, value);
 * }</pre>
 */
public class MultiHeadAttention extends NNModule {

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final float scale;

    private final Linear qProj;
    private final Linear kProj;
    private final Linear vProj;
    private final Linear outProj;

    /**
     * Create a multi-head attention module.
     *
     * @param embedDim dimension of embeddings (must be divisible by numHeads)
     * @param numHeads number of attention heads
     *
     * @throws IllegalArgumentException if embedDim is not divisible by numHeads
     */
    public MultiHeadAttention(int embedDim, int numHeads) {
        this(embedDim, numHeads, 0.0f);
    }

    /**
     * Create a multi-head attention module with optional dropout.
     *
     * @param embedDim dimension of embeddings (must be divisible by numHeads)
     * @param numHeads number of attention heads
     * @param dropoutP dropout probability (0.0 = no dropout)
     *
     * @throws IllegalArgumentException if embedDim is not divisible by numHeads
     */
    public MultiHeadAttention(int embedDim, int numHeads, float dropoutP) {
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException(
                "embedDim (" + embedDim + ") must be divisible by numHeads (" + numHeads + ")");
        }
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be positive, got: " + numHeads);
        }
        if (dropoutP < 0 || dropoutP > 1) {
            throw new IllegalArgumentException("dropout probability must be in [0, 1], got: " + dropoutP);
        }

        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.scale = (float) (1.0 / Math.sqrt(headDim));

        this.qProj = register("q_proj", new Linear(embedDim, embedDim));
        this.kProj = register("k_proj", new Linear(embedDim, embedDim));
        this.vProj = register("v_proj", new Linear(embedDim, embedDim));
        this.outProj = register("out_proj", new Linear(embedDim, embedDim));
    }

    /**
     * Self-attention: query = key = value = input.
     *
     * @param input [batch, seq, embedDim]
     * @return [batch, seq, embedDim]
     */
    @Override
    public GradTensor forward(GradTensor input) {
        return forward(input, input, input, false);
    }

    /**
     * Self-attention with optional causal masking for autoregressive models.
     *
     * @param input      [batch, seq, embedDim]
     * @param causalMask if true, prevent attention to future positions
     * @return [batch, seq, embedDim]
     */
    public GradTensor forward(GradTensor input, boolean causalMask) {
        return forward(input, input, input, causalMask);
    }

    /**
     * Full attention with separate Q, K, V and optional causal masking.
     *
     * @param query       [batch, seq_q, embedDim]
     * @param key         [batch, seq_k, embedDim]
     * @param value       [batch, seq_k, embedDim]
     * @param causalMask  if true, prevent attention to future positions (for autoregressive models)
     * @return            [batch, seq_q, embedDim]
     *
     * @throws IllegalArgumentException if dimensions are incompatible
     */
    public GradTensor forward(GradTensor query, GradTensor key, GradTensor value, boolean causalMask) {
        return forward(query, key, value, causalMask ? createCausalMask(query.shape()[1], key.shape()[1]) : null);
    }

    /**
     * Full attention with separate Q, K, V and optional mask.
     *
     * @param query  [batch, seq_q, embedDim] query embeddings
     * @param key    [batch, seq_k, embedDim] key embeddings
     * @param value  [batch, seq_k, embedDim] value embeddings
     * @param mask   optional attention mask where 0 = masked position, 1 = attend (null = no mask)
     * @return       [batch, seq_q, embedDim] attention output
     *
     * @throws IllegalArgumentException if tensor dimensions are incompatible
     */
    public GradTensor forward(GradTensor query, GradTensor key, GradTensor value, GradTensor mask) {
        long[] qs = query.shape();
        long[] ks = key.shape();
        long[] vs = value.shape();

        // Validate input shapes
        if (qs.length != 3 || ks.length != 3 || vs.length != 3) {
            throw new IllegalArgumentException(
                "Expected 3D tensors [batch, seq, dim], got shapes: " +
                java.util.Arrays.toString(qs) + ", " + java.util.Arrays.toString(ks) +
                ", " + java.util.Arrays.toString(vs));
        }
        if (qs[0] != ks[0] || qs[0] != vs[0]) {
            throw new IllegalArgumentException(
                "batch size mismatch: query=" + qs[0] + ", key=" + ks[0] + ", value=" + vs[0]);
        }
        if (qs[2] != embedDim || ks[2] != embedDim || vs[2] != embedDim) {
            throw new IllegalArgumentException(
                "embedding dimension mismatch: expected " + embedDim + ", got query=" + qs[2] +
                ", key=" + ks[2] + ", value=" + vs[2]);
        }
        if (ks[1] != vs[1]) {
            throw new IllegalArgumentException(
                "key and value sequence lengths must match: key=" + ks[1] + ", value=" + vs[1]);
        }

        int batch = (int) qs[0];
        int seqQ = (int) qs[1];
        int seqK = (int) ks[1];

        // Project Q, K, V
        GradTensor q = qProj.forward(query);  // [batch, seqQ, embedDim]
        GradTensor k = kProj.forward(key);    // [batch, seqK, embedDim]
        GradTensor v = vProj.forward(value);  // [batch, seqK, embedDim]

        // Reshape to multi-head: [batch * numHeads, seq, headDim]
        q = reshapeToHeads(q, batch, seqQ);
        k = reshapeToHeads(k, batch, seqK);
        v = reshapeToHeads(v, batch, seqK);

        // Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
        GradTensor attnWeights = q.matmul(k.transpose()).mul(scale);
        // [batch*numHeads, seqQ, seqK]

        // Apply mask if provided
        if (mask != null) {
            attnWeights = applyMask(attnWeights, mask);
        }

        attnWeights = attnWeights.softmax();

        // Attend to values
        GradTensor attnOutput = attnWeights.matmul(v);  // [batch*numHeads, seqQ, headDim]

        // Reshape back: [batch, seqQ, embedDim]
        attnOutput = reshapeFromHeads(attnOutput, batch, seqQ);

        // Final projection
        return outProj.forward(attnOutput);
    }

    /**
     * Create a causal attention mask for autoregressive generation.
     * <p>
     * Prevents the model from attending to future positions. The returned mask
     * is lower triangular: position i can only attend to positions 0..i.
     *
     * @param seqQ query sequence length
     * @param seqK key/value sequence length
     * @return mask tensor where 1 = attend, 0 = mask, shape [seqQ, seqK]
     */
    public static GradTensor createCausalMask(long seqQ, long seqK) {
        float[] mask = new float[(int)(seqQ * seqK)];
        for (int i = 0; i < seqQ; i++) {
            for (int j = 0; j < seqK; j++) {
                // Can attend to current and past positions only
                mask[i * (int)seqK + j] = (j <= i) ? 1.0f : 0.0f;
            }
        }
        return GradTensor.of(mask, seqQ, seqK);
    }

    /**
     * Reshape from [batch, seq, embedDim] to [batch*numHeads, seq, headDim].
     * This distributes the embedDim across multiple heads.
     */
    private GradTensor reshapeToHeads(GradTensor x, int batch, int seq) {
        float[] data = x.data();
        float[] result = new float[batch * numHeads * seq * headDim];

        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                for (int h = 0; h < numHeads; h++) {
                    int srcOff = b * seq * embedDim + s * embedDim + h * headDim;
                    int dstOff = (b * numHeads + h) * seq * headDim + s * headDim;
                    System.arraycopy(data, srcOff, result, dstOff, headDim);
                }
            }
        }
        return GradTensor.of(result, batch * numHeads, seq, headDim)
                .requiresGrad(x.requiresGrad());
    }

    /**
     * Reshape from [batch*numHeads, seq, headDim] back to [batch, seq, embedDim].
     * This recombines the heads back into a single embedding dimension.
     */
    private GradTensor reshapeFromHeads(GradTensor x, int batch, int seq) {
        float[] data = x.data();
        float[] result = new float[batch * seq * embedDim];

        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                for (int h = 0; h < numHeads; h++) {
                    int srcOff = (b * numHeads + h) * seq * headDim + s * headDim;
                    int dstOff = b * seq * embedDim + s * embedDim + h * headDim;
                    System.arraycopy(data, srcOff, result, dstOff, headDim);
                }
            }
        }
        return GradTensor.of(result, batch, seq, embedDim)
                .requiresGrad(x.requiresGrad());
    }

    /**
     * Apply attention mask to attention weights.
     * <p>
     * Positions where mask is 0 are set to a large negative value so they
     * become near-zero after softmax. Positions where mask is 1 are left unchanged.
     *
     * @param attnWeights attention weight matrix [batch*heads, seq_q, seq_k]
     * @param mask        attention mask [seq_q, seq_k] where 0 = mask, 1 = attend
     * @return            masked attention weights
     */
    private GradTensor applyMask(GradTensor attnWeights, GradTensor mask) {
        float[] aw = attnWeights.data();
        float[] m = mask.data();
        float[] result = new float[aw.length];

        long[] awShape = attnWeights.shape();
        int seqQ = (int) awShape[1];
        int seqK = (int) awShape[2];

        // Apply mask to each head independently
        for (int i = 0; i < aw.length; i++) {
            // Map flat index to (seq_q, seq_k) position
            int posInHead = i % (seqQ * seqK);
            int maskIdx = posInHead;
            result[i] = m[maskIdx] > 0.5f ? aw[i] : -1e9f;
        }

        return GradTensor.of(result, attnWeights.shape())
                .requiresGrad(attnWeights.requiresGrad());
    }

    /**
     * Get the embedding dimension.
     *
     * @return embedDim
     */
    public int getEmbedDim() {
        return embedDim;
    }

    /**
     * Get the number of attention heads.
     *
     * @return numHeads
     */
    public int getNumHeads() {
        return numHeads;
    }

    @Override
    public String toString() {
        return "MultiHeadAttention(embed=" + embedDim + ", heads=" + numHeads + ")";
    }
}
