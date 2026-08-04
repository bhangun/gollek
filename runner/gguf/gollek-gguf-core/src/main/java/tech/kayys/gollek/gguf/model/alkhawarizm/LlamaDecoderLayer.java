package tech.kayys.gollek.gguf.model.alkhawarizm;

import tech.kayys.alkhawarizm.core.nn.Module;
import tech.kayys.alkhawarizm.core.nn.RMSNorm;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.WeightAdapter;
import tech.kayys.gollek.gguf.model.ModelConfig;

public class LlamaDecoderLayer extends Module {
    private final LlamaAttention selfAttn;
    private final LlamaMLP mlp;
    private final RMSNorm inputLayerNorm;
    private final RMSNorm postAttentionLayerNorm;

    public LlamaDecoderLayer(ModelConfig cfg, WeightAdapter weights, int layerIdx) {
        float eps = cfg.rmsNormEps();
        int dim = cfg.embeddingDim();

        inputLayerNorm = new RMSNorm(weights.getWeight("blk." + layerIdx + ".attn_norm.weight"), eps);
        selfAttn = new LlamaAttention(cfg, weights, layerIdx);

        postAttentionLayerNorm = new RMSNorm(weights.getWeight("blk." + layerIdx + ".ffn_norm.weight"), eps);
        mlp = new LlamaMLP(weights, layerIdx);

        registerModule("input_layernorm", inputLayerNorm);
        registerModule("self_attn", selfAttn);
        registerModule("post_attention_layernorm", postAttentionLayerNorm);
        registerModule("mlp", mlp);
    }

    @Override
    public Tensor forward(Tensor x) {
        // Attention block
        Tensor residual = x;
        x = inputLayerNorm.forward(x);
        x = selfAttn.forward(x);
        x = x.add(residual);

        // MLP block
        residual = x;
        x = postAttentionLayerNorm.forward(x);
        x = mlp.forward(x);
        x = x.add(residual);

        return x;
    }
}
