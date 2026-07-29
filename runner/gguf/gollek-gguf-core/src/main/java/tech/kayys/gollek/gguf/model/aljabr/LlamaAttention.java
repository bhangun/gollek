package tech.kayys.gollek.gguf.model.alkhawarizm;

import tech.kayys.alkhawarizm.core.nn.Linear;
import tech.kayys.alkhawarizm.core.nn.Module;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.TensorFactory;
import tech.kayys.alkhawarizm.core.tensor.WeightAdapter;
import tech.kayys.gollek.gguf.model.ModelConfig;

public class LlamaAttention extends Module {
    private final int nHeads;
    private final int nKVHeads;
    private final int headDim;
    private final int layerIdx;
    private final ModelConfig cfg;

    private final Linear qProj;
    private final Linear kProj;
    private final Linear vProj;
    private final Linear oProj;

    public LlamaAttention(ModelConfig cfg, WeightAdapter weights, int layerIdx) {
        this.cfg = cfg;
        this.layerIdx = layerIdx;
        this.nHeads = cfg.nHeads();
        // Load weights from adapter
        Tensor wQ = weights.getWeight("blk." + layerIdx + ".attn_q.weight");

        Tensor wK = null;
        for (int i = layerIdx; i >= 0; i--) {
            wK = weights.getWeight("blk." + i + ".attn_k.weight");
            if (wK != null)
                break;
        }

        this.headDim = wQ != null ? (int) (wQ.shape().dim(0) / nHeads) : cfg.headDim();
        this.nKVHeads = wK != null ? (int) (wK.shape().dim(0) / this.headDim) : cfg.nKVHeads();

        Tensor wV = null;
        for (int i = layerIdx; i >= 0; i--) {
            wV = weights.getWeight("blk." + i + ".attn_v.weight");
            if (wV != null)
                break;
        }

        Tensor wO = weights.getWeight("blk." + layerIdx + ".attn_output.weight");

        qProj = new Linear(wQ, null);
        kProj = new Linear(wK, null);
        vProj = new Linear(wV, null);
        oProj = new Linear(wO, null);

        registerModule("q_proj", qProj);
        registerModule("k_proj", kProj);
        registerModule("v_proj", vProj);
        registerModule("o_proj", oProj);
    }

    @Override
    public Tensor forward(Tensor x) {
        // x shape: [batch, seq_len, hidden_size]
        long batch = x.shape().dim(0);
        long seqLen = x.shape().dim(1);

        Tensor q = qProj.forward(x);
        Tensor k = kProj.forward(x);
        Tensor v = vProj.forward(x);

        System.err.println("q out shape: " + q.shape());
        System.err.println("k out shape: " + k.shape());
        System.err.println("v out shape: " + v.shape());
        System.err.println("Reshaping q to: batch=" + batch + ", seqLen=" + seqLen + ", nHeads=" + nHeads + ", headDim="
                + headDim);
        System.err.println("Reshaping k to: batch=" + batch + ", seqLen=" + seqLen + ", nKVHeads=" + nKVHeads
                + ", headDim=" + headDim);

        q = q.reshape(batch, seqLen, nHeads, headDim); // [B, S, H, D]
        k = k.reshape(batch, seqLen, nKVHeads, headDim); // [B, S, nKVH, D]
        v = v.reshape(batch, seqLen, nKVHeads, headDim); // [B, S, nKVH, D]

        // Apply RoPE
        q = q.applyRoPE(headDim, cfg.ropeFreqBase(), true);
        k = k.applyRoPE(headDim, cfg.ropeFreqBase(), true);

        // K/V cache update
        Tensor cachedK = k.updateKVCache(layerIdx, "k");
        Tensor cachedV = v.updateKVCache(layerIdx, "v");

        // Native attention execution
        Tensor out = q.attention(cachedK, cachedV);

        // Flatten heads for output projection: [B, S, H * D]
        out = out.reshape(batch, seqLen, (long) nHeads * headDim);

        return oProj.forward(out);
    }
}
