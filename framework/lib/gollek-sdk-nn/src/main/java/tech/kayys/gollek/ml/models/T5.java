package tech.kayys.gollek.ml.models;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.TensorOps;
import tech.kayys.gollek.ml.nn.*;

/**
 * T5 (Text-to-Text Transfer Transformer) — encoder-decoder architecture
 * for sequence-to-sequence tasks.
 *
 * <p>Based on <em>"Exploring the Limits of Transfer Learning with a Unified
 * Text-to-Text Transformer"</em> (Raffel et al., 2019).
 *
 * <p>Key differences from BERT/GPT:
 * <ul>
 *   <li>Encoder-decoder (not encoder-only or decoder-only)</li>
 *   <li>Relative position bias instead of absolute positional encoding</li>
 *   <li>Pre-norm with RMSNorm (same as LLaMA)</li>
 *   <li>No bias in any layer</li>
 * </ul>
 *
 * <p>Available variants:
 * <ul>
 *   <li>{@link #t5Small(int)} — 6 layers, 512 hidden, 8 heads, ~60M params</li>
 *   <li>{@link #t5Tiny(int)} — 2 layers, 128 hidden, 4 heads (for testing)</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * NNModule t5 = T5.t5Small(vocabSize = 32128);
 * // encoder input + decoder input → decoder output logits
 * GradTensor logits = ((T5.T5Model) t5).forward(encInput, decInput);
 * }</pre>
 */
public final class T5 {

    private T5() {}

    /**
     * T5-Small: 6 encoder + 6 decoder layers, 512 hidden, 8 heads.
     *
     * @param vocabSize vocabulary size (32128 for standard T5)
     * @return T5-Small model
     */
    public static T5Model t5Small(int vocabSize) {
        return new T5Model(512, 8, 6, 2048, vocabSize);
    }

    /**
     * T5-Tiny: 2 layers, 128 hidden, 4 heads (for testing).
     *
     * @param vocabSize vocabulary size
     * @return T5-Tiny model
     */
    public static T5Model t5Tiny(int vocabSize) {
        return new T5Model(128, 4, 2, 512, vocabSize);
    }

    // ── T5 Model ──────────────────────────────────────────────────────────

    /**
     * T5 encoder-decoder model.
     */
    public static final class T5Model extends NNModule {

        private final Embedding sharedEmbed;
        private final T5Block[] encoderBlocks;
        private final T5Block[] decoderBlocks;
        private final LLaMA.RMSNorm encNorm, decNorm;
        private final Linear lmHead;

        T5Model(int dModel, int nHeads, int nLayers, int dFF, int vocabSize) {
            this.sharedEmbed  = register("embed",    new Embedding(vocabSize, dModel));
            this.encoderBlocks = new T5Block[nLayers];
            this.decoderBlocks = new T5Block[nLayers];
            for (int i = 0; i < nLayers; i++) {
                encoderBlocks[i] = register("enc_" + i, new T5Block(dModel, nHeads, dFF, false));
                decoderBlocks[i] = register("dec_" + i, new T5Block(dModel, nHeads, dFF, true));
            }
            this.encNorm = register("enc_norm", new LLaMA.RMSNorm(dModel));
            this.decNorm = register("dec_norm", new LLaMA.RMSNorm(dModel));
            this.lmHead  = register("lm_head",  new Linear(dModel, vocabSize, false));
        }

        /**
         * Forward pass through encoder then decoder.
         *
         * @param encInput encoder input embeddings {@code [B, Ts, dModel]}
         * @param decInput decoder input embeddings {@code [B, Tt, dModel]}
         * @return decoder logits {@code [B, Tt, vocabSize]}
         */
        public GradTensor forward(GradTensor encInput, GradTensor decInput) {
            // Encode
            GradTensor enc = encInput;
            for (T5Block block : encoderBlocks) enc = block.forward(enc, null);
            enc = encNorm.forward(enc);

            // Decode with cross-attention to encoder output
            GradTensor dec = decInput;
            for (T5Block block : decoderBlocks) dec = block.forward(dec, enc);
            dec = decNorm.forward(dec);

            return lmHead.forward(dec);
        }

        @Override
        public GradTensor forward(GradTensor x) { return forward(x, x); }
    }

    // ── T5 Block ──────────────────────────────────────────────────────────

    /**
     * T5 transformer block with optional cross-attention (for decoder).
     */
    static final class T5Block extends NNModule {

        private final LLaMA.RMSNorm norm1, norm2, norm3;
        private final Linear wQ, wK, wV, wO;           // self-attention
        private final Linear xwQ, xwK, xwV, xwO;       // cross-attention (decoder only)
        private final Linear ff1, ff2;
        private final int nHeads, headDim;
        private final boolean isDecoder;

        T5Block(int dModel, int nHeads, int dFF, boolean isDecoder) {
            this.nHeads    = nHeads;
            this.headDim   = dModel / nHeads;
            this.isDecoder = isDecoder;

            this.norm1 = register("norm1", new LLaMA.RMSNorm(dModel));
            this.norm2 = register("norm2", new LLaMA.RMSNorm(dModel));
            this.wQ    = register("wQ",    new Linear(dModel, dModel, false));
            this.wK    = register("wK",    new Linear(dModel, dModel, false));
            this.wV    = register("wV",    new Linear(dModel, dModel, false));
            this.wO    = register("wO",    new Linear(dModel, dModel, false));
            this.ff1   = register("ff1",   new Linear(dModel, dFF, false));
            this.ff2   = register("ff2",   new Linear(dFF, dModel, false));

            if (isDecoder) {
                this.norm3 = register("norm3", new LLaMA.RMSNorm(dModel));
                this.xwQ   = register("xwQ",   new Linear(dModel, dModel, false));
                this.xwK   = register("xwK",   new Linear(dModel, dModel, false));
                this.xwV   = register("xwV",   new Linear(dModel, dModel, false));
                this.xwO   = register("xwO",   new Linear(dModel, dModel, false));
            } else {
                this.norm3 = null; this.xwQ = null; this.xwK = null;
                this.xwV = null;   this.xwO = null;
            }
        }

        GradTensor forward(GradTensor x, GradTensor encOut) {
            // Self-attention
            x = x.add(attention(norm1.forward(x), null, false));
            // Cross-attention (decoder only)
            if (isDecoder && encOut != null)
                x = x.add(attention(norm3.forward(x), encOut, true));
            // FFN
            x = x.add(ff2.forward(ff1.forward(norm2.forward(x)).relu()));
            return x;
        }

        @Override public GradTensor forward(GradTensor x) { return forward(x, null); }

        private GradTensor attention(GradTensor q, GradTensor kv, boolean cross) {
            long[] s = q.shape();
            int B = (int)s[0], T = (int)s[1], D = (int)s[2];
            float scale = (float)(1.0 / Math.sqrt(headDim));

            GradTensor K = cross ? xwK.forward(kv) : wK.forward(q);
            GradTensor V = cross ? xwV.forward(kv) : wV.forward(q);
            GradTensor Q = cross ? xwQ.forward(q)  : wQ.forward(q);

            GradTensor scores = TensorOps.einsum("bhid,bhjd->bhij",
                reshape4d(Q, B, T, nHeads, headDim),
                reshape4d(K, B, (int)K.shape()[1], nHeads, headDim)).mul(scale);
            GradTensor attn = scores.softmax();
            GradTensor out  = TensorOps.einsum("bhij,bhjd->bhid", attn,
                reshape4d(V, B, (int)V.shape()[1], nHeads, headDim));

            Linear outProj = cross ? xwO : wO;
            return outProj.forward(out.reshape(B, T, D));
        }

        private static GradTensor reshape4d(GradTensor x, int B, int T, int H, int D) {
            float[] src = x.data(), dst = new float[B*H*T*D];
            for (int b=0;b<B;b++) for (int t=0;t<T;t++) for (int h=0;h<H;h++) for (int d=0;d<D;d++)
                dst[b*H*T*D+h*T*D+t*D+d] = src[b*T*H*D+t*H*D+h*D+d];
            return GradTensor.of(dst, B, H, T, D);
        }
    }
}
