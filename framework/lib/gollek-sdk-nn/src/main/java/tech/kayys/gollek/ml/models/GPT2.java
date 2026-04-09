package tech.kayys.gollek.ml.models;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;

/**
 * GPT-2 style autoregressive language model.
 *
 * <p>Decoder-only Transformer with causal (masked) self-attention.
 * Based on <em>"Language Models are Unsupervised Multitask Learners"</em>
 * (Radford et al., 2019).
 *
 * <p>Available variants:
 * <ul>
 *   <li>{@link #gpt2Small(int)} — 12 layers, 768 hidden, 12 heads, ~117M params</li>
 *   <li>{@link #gpt2Medium(int)} — 24 layers, 1024 hidden, 16 heads, ~345M params</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * NNModule gpt = GPT2.gpt2Small(vocabSize = 50257);
 * GradTensor logits = gpt.forward(embeddings); // [B, T, 768] → [B, T, vocabSize]
 * }</pre>
 */
public final class GPT2 {

    private GPT2() {}

    /**
     * GPT-2 Small: 12 layers, dModel=768, 12 heads, dFF=3072.
     *
     * @param vocabSize vocabulary size (50257 for standard GPT-2)
     * @return GPT-2 Small model
     */
    public static NNModule gpt2Small(int vocabSize) {
        return new GPT2Model(768, 12, 12, 3072, 1024, vocabSize, 0.1f);
    }

    /**
     * GPT-2 Medium: 24 layers, dModel=1024, 16 heads, dFF=4096.
     *
     * @param vocabSize vocabulary size
     * @return GPT-2 Medium model
     */
    public static NNModule gpt2Medium(int vocabSize) {
        return new GPT2Model(1024, 16, 24, 4096, 1024, vocabSize, 0.1f);
    }

    // ── Internal model ────────────────────────────────────────────────────

    static final class GPT2Model extends NNModule {

        private final Embedding tokenEmbed;
        private final PositionalEncoding posEnc;
        private final TransformerBlock[] blocks;
        private final LayerNorm finalNorm;
        private final Linear lmHead;   // language model head: dModel → vocabSize

        /**
         * @param dModel    model dimension
         * @param nHeads    attention heads
         * @param nLayers   decoder blocks
         * @param dFF       feed-forward hidden size
         * @param maxSeqLen maximum sequence length
         * @param vocabSize vocabulary size
         * @param dropout   dropout rate
         */
        GPT2Model(int dModel, int nHeads, int nLayers, int dFF,
                  int maxSeqLen, int vocabSize, float dropout) {
            this.tokenEmbed = register("token_embed", new Embedding(vocabSize, dModel));
            this.posEnc     = register("pos_enc",     new PositionalEncoding(dModel, maxSeqLen));
            this.blocks     = new TransformerBlock[nLayers];
            for (int i = 0; i < nLayers; i++)
                blocks[i] = register("block_" + i, new TransformerBlock(dModel, nHeads, dFF, dropout));
            this.finalNorm  = register("norm",    new LayerNorm(dModel));
            this.lmHead     = register("lm_head", new Linear(dModel, vocabSize, false));
        }

        /**
         * Forward pass through the GPT-2 decoder.
         *
         * @param x input embeddings {@code [B, T, dModel]}
         * @return logits over vocabulary {@code [B, T, vocabSize]}
         */
        @Override
        public GradTensor forward(GradTensor x) {
            x = posEnc.forward(x);
            for (TransformerBlock block : blocks) x = block.forward(x);
            x = finalNorm.forward(x);
            return lmHead.forward(x);
        }
    }
}
