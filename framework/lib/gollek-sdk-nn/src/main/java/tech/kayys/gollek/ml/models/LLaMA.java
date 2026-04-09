package tech.kayys.gollek.ml.models;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;

/**
 * LLaMA-style decoder-only language model.
 *
 * <p>Implements the architecture from <em>"LLaMA: Open and Efficient Foundation
 * Language Models"</em> (Touvron et al., 2023) with key improvements over GPT-2:
 * <ul>
 *   <li>RMSNorm instead of LayerNorm (pre-norm)</li>
 *   <li>SwiGLU feed-forward network</li>
 *   <li>Rotary Position Embeddings (RoPE) instead of learned positional encoding</li>
 *   <li>No bias in attention projections</li>
 * </ul>
 *
 * <p>Available variants:
 * <ul>
 *   <li>{@link #llama7B(int)} — 32 layers, 4096 hidden, 32 heads, ~7B params</li>
 *   <li>{@link #llamaTiny(int)} — 4 layers, 256 hidden, 4 heads, ~few M params (for testing)</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * NNModule llama = LLaMA.llamaTiny(vocabSize = 32000);
 * GradTensor logits = llama.forward(embeddings); // [B, T, D] → [B, T, vocabSize]
 * }</pre>
 */
public final class LLaMA {

    private LLaMA() {}

    /**
     * LLaMA-7B configuration: 32 layers, 4096 hidden, 32 heads.
     * Note: full 7B model requires ~28GB RAM. Use for architecture reference.
     *
     * @param vocabSize vocabulary size (32000 for LLaMA tokenizer)
     * @return LLaMA-7B model
     */
    public static NNModule llama7B(int vocabSize) {
        return new LLaMAModel(4096, 32, 32, 11008, 2048, vocabSize, 0.0f);
    }

    /**
     * Tiny LLaMA for testing: 4 layers, 256 hidden, 4 heads, ~few M params.
     *
     * @param vocabSize vocabulary size
     * @return tiny LLaMA model
     */
    public static NNModule llamaTiny(int vocabSize) {
        return new LLaMAModel(256, 4, 4, 1024, 512, vocabSize, 0.0f);
    }

    // ── Internal model ────────────────────────────────────────────────────

    static final class LLaMAModel extends NNModule {

        private final Embedding tokenEmbed;
        private final LLaMABlock[] blocks;
        private final RMSNorm finalNorm;
        private final Linear lmHead;

        LLaMAModel(int dModel, int nHeads, int nLayers, int dFF,
                   int maxSeqLen, int vocabSize, float dropout) {
            this.tokenEmbed = register("embed",    new Embedding(vocabSize, dModel));
            this.blocks     = new LLaMABlock[nLayers];
            for (int i = 0; i < nLayers; i++)
                blocks[i] = register("block_" + i,
                    new LLaMABlock(dModel, nHeads, dFF, maxSeqLen));
            this.finalNorm  = register("norm",     new RMSNorm(dModel));
            this.lmHead     = register("lm_head",  new Linear(dModel, vocabSize, false));
        }

        @Override
        public GradTensor forward(GradTensor x) {
            for (LLaMABlock block : blocks) x = block.forward(x);
            return lmHead.forward(finalNorm.forward(x));
        }
    }

    // ── LLaMA decoder block ───────────────────────────────────────────────

    /**
     * Single LLaMA decoder block: RMSNorm → Attention (RoPE) → RMSNorm → SwiGLU FFN.
     */
    static final class LLaMABlock extends NNModule {

        private final RMSNorm  norm1, norm2;
        private final Linear   wQ, wK, wV, wO;
        private final Linear   wGate, wUp, wDown; // SwiGLU FFN
        private final RotaryEmbedding rope;
        private final int nHeads, headDim;

        LLaMABlock(int dModel, int nHeads, int dFF, int maxSeqLen) {
            this.nHeads  = nHeads;
            this.headDim = dModel / nHeads;
            this.norm1   = register("norm1", new RMSNorm(dModel));
            this.norm2   = register("norm2", new RMSNorm(dModel));
            this.wQ      = register("wQ",    new Linear(dModel, dModel, false));
            this.wK      = register("wK",    new Linear(dModel, dModel, false));
            this.wV      = register("wV",    new Linear(dModel, dModel, false));
            this.wO      = register("wO",    new Linear(dModel, dModel, false));
            this.wGate   = register("wGate", new Linear(dModel, dFF, false));
            this.wUp     = register("wUp",   new Linear(dModel, dFF, false));
            this.wDown   = register("wDown", new Linear(dFF, dModel, false));
            this.rope    = register("rope",  new RotaryEmbedding(headDim, maxSeqLen));
        }

        @Override
        public GradTensor forward(GradTensor x) {
            // Pre-norm attention with residual
            GradTensor normed = norm1.forward(x);
            x = x.add(attention(normed));
            // Pre-norm SwiGLU FFN with residual
            x = x.add(swiglu(norm2.forward(x)));
            return x;
        }

        private GradTensor attention(GradTensor x) {
            long[] s = x.shape();
            int B = (int)s[0], T = (int)s[1], D = (int)s[2];
            float scale = (float)(1.0 / Math.sqrt(headDim));

            GradTensor Q = reshape4d(wQ.forward(x), B, T, nHeads, headDim);
            GradTensor K = reshape4d(wK.forward(x), B, T, nHeads, headDim);
            GradTensor V = reshape4d(wV.forward(x), B, T, nHeads, headDim);

            // Apply RoPE to Q and K
            Q = rope.apply(Q);
            K = rope.apply(K);

            // Scaled dot-product attention (causal)
            GradTensor scores = tech.kayys.gollek.ml.autograd.TensorOps
                .einsum("bhid,bhjd->bhij", Q, K).mul(scale);
            // Apply causal mask
            scores = applyCausalMask(scores, T);
            GradTensor attn = scores.softmax();
            GradTensor out  = tech.kayys.gollek.ml.autograd.TensorOps
                .einsum("bhij,bhjd->bhid", attn, V);
            return wO.forward(out.reshape(B, T, D));
        }

        /** SwiGLU: down(silu(gate(x)) * up(x)) */
        private GradTensor swiglu(GradTensor x) {
            GradTensor gate = wGate.forward(x).silu();
            GradTensor up   = wUp.forward(x);
            return wDown.forward(gate.mul(up));
        }

        private static GradTensor applyCausalMask(GradTensor scores, int T) {
            float[] d = scores.data().clone();
            long[] s = scores.shape();
            int B = (int)s[0], H = (int)s[1];
            for (int b = 0; b < B; b++)
                for (int h = 0; h < H; h++)
                    for (int i = 0; i < T; i++)
                        for (int j = i+1; j < T; j++)
                            d[b*H*T*T + h*T*T + i*T + j] = Float.NEGATIVE_INFINITY;
            return GradTensor.of(d, s);
        }

        private static GradTensor reshape4d(GradTensor x, int B, int T, int H, int D) {
            float[] src = x.data(), dst = new float[B*H*T*D];
            for (int b=0;b<B;b++) for (int t=0;t<T;t++) for (int h=0;h<H;h++) for (int d=0;d<D;d++)
                dst[b*H*T*D+h*T*D+t*D+d] = src[b*T*H*D+t*H*D+h*D+d];
            return GradTensor.of(dst, B, H, T, D);
        }
    }

    // ── RMSNorm ───────────────────────────────────────────────────────────

    /**
     * Root Mean Square Layer Normalization — faster than LayerNorm (no mean subtraction).
     * Used in LLaMA, T5, and other modern LLMs.
     */
    static final class RMSNorm extends NNModule {
        private final Parameter weight;
        private final float eps;

        RMSNorm(int dim) { this(dim, 1e-6f); }

        RMSNorm(int dim, float eps) {
            this.eps    = eps;
            float[] w   = new float[dim]; java.util.Arrays.fill(w, 1f);
            this.weight = registerParameter("weight", GradTensor.of(w, dim));
        }

        @Override
        public GradTensor forward(GradTensor x) {
            long[] s = x.shape();
            int last = (int) s[s.length - 1];
            float[] d = x.data(), w = weight.data().data();
            float[] out = new float[d.length];
            int outer = d.length / last;
            for (int i = 0; i < outer; i++) {
                float sumSq = 0f;
                for (int j = 0; j < last; j++) sumSq += d[i*last+j] * d[i*last+j];
                float rms = (float) Math.sqrt(sumSq / last + eps);
                for (int j = 0; j < last; j++) out[i*last+j] = w[j] * d[i*last+j] / rms;
            }
            return GradTensor.of(out, s);
        }
    }
}
