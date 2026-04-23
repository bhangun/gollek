package tech.kayys.gollek.safetensor.runner.sd;

import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import java.util.Map;

/**
 * CLIP Text Encoder implementation for Stable Diffusion.
 */
public class CLIPModel {
    private final Map<String, AccelTensor> weights;

    public CLIPModel(Map<String, AccelTensor> weights) {
        this.weights = weights;
    }

    public AccelTensor encode(long[] inputIds) {
        // 1. Embeddings
        AccelTensor tokenEmbeds = weights.get("text_model.embeddings.token_embedding.weight");
        AccelTensor posEmbeds = weights.get("text_model.embeddings.position_embedding.weight");
        
        // Lookup
        AccelTensor x = tokenEmbeds.indexSelect(inputIds);
        
        // Add positions
        long[] posIds = new long[inputIds.length];
        for (int i = 0; i < inputIds.length; i++) posIds[i] = i;
        AccelTensor posX = posEmbeds.indexSelect(posIds);
        
        AccelTensor hidden = AccelOps.add(x, posX);
        
        // 2. Layers
        for (int i = 0; i < 12; i++) {
            hidden = layer(hidden, i);
        }
        
        // 3. Final Layer Norm
        AccelTensor lnW = weights.get("text_model.final_layer_norm.weight");
        AccelTensor lnB = weights.get("text_model.final_layer_norm.bias");
        return AccelOps.layerNorm(hidden, lnW, lnB, 1e-5);
    }

    private AccelTensor layer(AccelTensor x, int i) {
        String base = "text_model.encoder.layers." + i + ".";
        
        // Self Attention
        AccelTensor residual = x;
        AccelTensor ln1W = weights.get(base + "layer_norm1.weight");
        AccelTensor ln1B = weights.get(base + "layer_norm1.bias");
        AccelTensor normX = AccelOps.layerNorm(x, ln1W, ln1B, 1e-5);
        
        AccelTensor attnOut = attention(normX, base + "self_attn.");
        x = AccelOps.add(residual, attnOut);
        
        // MLP
        residual = x;
        AccelTensor ln2W = weights.get(base + "layer_norm2.weight");
        AccelTensor ln2B = weights.get(base + "layer_norm2.bias");
        normX = AccelOps.layerNorm(x, ln2W, ln2B, 1e-5);
        
        AccelTensor mlpOut = mlp(normX, base + "mlp.");
        return AccelOps.add(residual, mlpOut);
    }

    private AccelTensor attention(AccelTensor x, String base) {
        AccelTensor qW = weights.get(base + "q_proj.weight");
        AccelTensor qB = weights.get(base + "q_proj.bias");
        AccelTensor kW = weights.get(base + "k_proj.weight");
        AccelTensor kB = weights.get(base + "k_proj.bias");
        AccelTensor vW = weights.get(base + "v_proj.weight");
        AccelTensor vB = weights.get(base + "v_proj.bias");
        AccelTensor outW = weights.get(base + "out_proj.weight");
        AccelTensor outB = weights.get(base + "out_proj.bias");

        long batch = x.size(0);
        long seq = x.size(1);
        long embed = x.size(2);
        long numHeads = 12;
        long headDim = embed / numHeads;

        // Linear projections: [B, S, E]
        AccelTensor q = AccelOps.add(AccelOps.linear(x, qW), qB);
        AccelTensor k = AccelOps.add(AccelOps.linear(x, kW), kB);
        AccelTensor v = AccelOps.add(AccelOps.linear(x, vW), vB);
        
        // Multi-head reshape: [B, S, H, D] -> [B, H, S, D]
        q = q.reshape(batch, seq, numHeads, headDim).transpose(1, 2);
        k = k.reshape(batch, seq, numHeads, headDim).transpose(1, 2);
        v = v.reshape(batch, seq, numHeads, headDim).transpose(1, 2);

        // Scaled Dot Product Attention
        float scale = (float) (1.0 / Math.sqrt(headDim));
        AccelTensor k_t = k.transpose(-2, -1);
        AccelTensor scores = AccelOps.mulScalar(AccelOps.matmul(q, k_t), scale);
        
        // Causal mask (optional for encoder, but CLIP often uses it)
        // For SD prompt encoding, usually it's full context but CLIP weights expect it.
        
        AccelTensor probs = AccelOps.softmax(scores, -1);
        AccelTensor context = AccelOps.matmul(probs, v); // [B, H, S, D]
        
        // Reshape back: [B, H, S, D] -> [B, S, E]
        AccelTensor out = context.transpose(1, 2).reshape(batch, seq, embed);
        
        return AccelOps.add(AccelOps.linear(out, outW), outB);
    }

    private AccelTensor mlp(AccelTensor x, String base) {
        AccelTensor fc1W = weights.get(base + "fc1.weight");
        AccelTensor fc1B = weights.get(base + "fc1.bias");
        AccelTensor fc2W = weights.get(base + "fc2.weight");
        AccelTensor fc2B = weights.get(base + "fc2.bias");
        
        AccelTensor h = AccelOps.add(AccelOps.linear(x, fc1W), fc1B);
        h = AccelOps.gelu(h);
        return AccelOps.add(AccelOps.linear(h, fc2W), fc2B);
    }
}
