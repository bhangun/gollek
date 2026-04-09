package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Transformer Encoder — stacks multiple {@link TransformerBlock}s with
 * an optional embedding layer and positional encoding.
 *
 * <p>Equivalent to {@code torch.nn.TransformerEncoder}.
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li>Input:  token IDs {@code [B, T]} (integer) or embeddings {@code [B, T, dModel]}</li>
 *   <li>Output: {@code [B, T, dModel]}</li>
 * </ul>
 *
 * <h3>Example — BERT-style encoder</h3>
 * <pre>{@code
 * var encoder = TransformerEncoder.builder()
 *     .vocabSize(30522)
 *     .dModel(768)
 *     .nHeads(12)
 *     .dFF(3072)
 *     .nLayers(12)
 *     .maxSeqLen(512)
 *     .dropout(0.1f)
 *     .build();
 *
 * GradTensor out = encoder.forward(embeddings); // [B, T, 768]
 * }</pre>
 */
public class TransformerEncoder extends NNModule {

    private final Embedding tokenEmbedding;
    private final PositionalEncoding posEncoding;
    private final TransformerBlock[] blocks;
    private final LayerNorm finalNorm;

    private TransformerEncoder(Builder b) {
        this.tokenEmbedding = b.vocabSize > 0
            ? register("embedding", new Embedding(b.vocabSize, b.dModel))
            : null;
        this.posEncoding = register("pos_enc",
            new PositionalEncoding(b.dModel, b.maxSeqLen));
        this.blocks = new TransformerBlock[b.nLayers];
        for (int i = 0; i < b.nLayers; i++) {
            blocks[i] = register("block_" + i,
                new TransformerBlock(b.dModel, b.nHeads, b.dFF, b.dropout));
        }
        this.finalNorm = register("norm", new LayerNorm(b.dModel));
    }

    /**
     * Forward pass through the full encoder stack.
     *
     * @param x input embeddings {@code [B, T, dModel]}
     * @return encoded representations {@code [B, T, dModel]}
     */
    @Override
    public GradTensor forward(GradTensor x) {
        x = posEncoding.forward(x);
        for (TransformerBlock block : blocks) x = block.forward(x);
        return finalNorm.forward(x);
    }

    /** @return number of stacked Transformer blocks */
    public int numLayers() { return blocks.length; }

    /**
     * Creates a new builder for {@link TransformerEncoder}.
     *
     * @return builder instance
     */
    public static Builder builder() { return new Builder(); }

    /**
     * Builder for {@link TransformerEncoder}.
     */
    public static final class Builder {
        private int vocabSize = 0;
        private int dModel    = 512;
        private int nHeads    = 8;
        private int dFF       = 2048;
        private int nLayers   = 6;
        private int maxSeqLen = 512;
        private float dropout = 0.1f;

        /** @param vocabSize vocabulary size (0 = skip embedding layer) */
        public Builder vocabSize(int v)  { this.vocabSize = v; return this; }
        /** @param dModel model dimension */
        public Builder dModel(int d)     { this.dModel = d; return this; }
        /** @param nHeads number of attention heads */
        public Builder nHeads(int n)     { this.nHeads = n; return this; }
        /** @param dFF feed-forward hidden size */
        public Builder dFF(int d)        { this.dFF = d; return this; }
        /** @param nLayers number of Transformer blocks */
        public Builder nLayers(int n)    { this.nLayers = n; return this; }
        /** @param maxSeqLen maximum sequence length for positional encoding */
        public Builder maxSeqLen(int m)  { this.maxSeqLen = m; return this; }
        /** @param dropout dropout probability */
        public Builder dropout(float d)  { this.dropout = d; return this; }

        /**
         * Builds the {@link TransformerEncoder}.
         *
         * @return configured encoder
         */
        public TransformerEncoder build() { return new TransformerEncoder(this); }
    }
}
