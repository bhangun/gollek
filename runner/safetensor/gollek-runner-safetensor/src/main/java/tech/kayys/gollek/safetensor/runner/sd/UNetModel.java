package tech.kayys.gollek.safetensor.runner.sd;

import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * UNet Denoiser implementation for Stable Diffusion.
 */
public class UNetModel {
    private final Map<String, AccelTensor> weights;

    public UNetModel(Map<String, AccelTensor> weights) {
        this.weights = weights;
    }

    public AccelTensor predict(AccelTensor sample, long timestep, AccelTensor encoderHiddenStates) {
        // 1. Initial Convolution
        AccelTensor x = AccelOps.conv2d(sample, weights.get("conv_in.weight"), weights.get("conv_in.bias"), 1, 1);
        
        // 2. Timestep Embedding
        AccelTensor tEmbed = getTimestepEmbedding(timestep);
        
        // 3. Down Blocks
        List<AccelTensor> residuals = new ArrayList<>();
        residuals.add(x); // Initial skip
        
        for (int i = 0; i < 4; i++) {
            String base = "down_blocks." + i + ".";
            // 2 ResNet blocks per down block
            for (int j = 0; j < 2; j++) {
                String resBase = base + "resnets." + j + ".";
                x = resnetBlock(x, tEmbed, resBase);
                if (weights.containsKey(base + "attentions." + j + ".norm.weight")) {
                    x = spatialTransformer(x, encoderHiddenStates, base + "attentions." + j + ".");
                }
                residuals.add(x);
            }
            // Downsample
            if (i < 3) {
                x = AccelOps.conv2d(x, weights.get(base + "downsamplers.0.conv.weight"), 
                                   weights.get(base + "downsamplers.0.conv.bias"), 2, 1);
                residuals.add(x);
            }
        }
        
        // 4. Mid Block
        x = resnetBlock(x, tEmbed, "mid_block.resnets.0.");
        x = spatialTransformer(x, encoderHiddenStates, "mid_block.attentions.0.");
        x = resnetBlock(x, tEmbed, "mid_block.resnets.1.");
        
        // 5. Up Blocks
        for (int i = 0; i < 4; i++) {
            String base = "up_blocks." + i + ".";
            for (int j = 0; j < 3; j++) {
                // Pop skip connection
                AccelTensor skip = residuals.remove(residuals.size() - 1);
                x = AccelOps.concat(x, skip, 1);
                
                String resBase = base + "resnets." + j + ".";
                x = resnetBlock(x, tEmbed, resBase);
                if (weights.containsKey(base + "attentions." + j + ".norm.weight")) {
                    x = spatialTransformer(x, encoderHiddenStates, base + "attentions." + j + ".");
                }
            }
            // Upsample
            if (i < 3) {
                x = upsample(x, base + "upsamplers.0.");
            }
        }
        
        // 6. Output
        AccelTensor norm = AccelOps.groupNorm(x, weights.get("conv_norm_out.weight"), weights.get("conv_norm_out.bias"), 32, 1e-5);
        x = AccelOps.silu(norm);
        return AccelOps.conv2d(x, weights.get("conv_out.weight"), weights.get("conv_out.bias"), 1, 1);
    }

    private AccelTensor upsample(AccelTensor x, String base) {
        // Nearest neighbor upsample 2x
        long b = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
        AccelTensor up = AccelTensor.zeros(b, c, h * 2, w * 2);
        // Direct copy for now (needs optimized kernel)
        for (int bi = 0; bi < b; bi++) {
            for (int ci = 0; ci < c; ci++) {
                for (int yi = 0; yi < h * 2; yi++) {
                    for (int xi = 0; xi < w * 2; xi++) {
                        up.set(x.get(bi, ci, yi / 2, xi / 2), bi, ci, yi, xi);
                    }
                }
            }
        }
        return AccelOps.conv2d(up, weights.get(base + "conv.weight"), weights.get(base + "conv.bias"), 1, 1);
    }

    private AccelTensor getTimestepEmbedding(long timestep) {
        int dim = 320;
        float[] sinEmb = new float[dim];
        double half = dim / 2.0;
        double logMax = Math.log(10000.0) / (half - 1);
        for (int i = 0; i < half; i++) {
            double freq = Math.exp(-i * logMax);
            double arg = timestep * freq;
            sinEmb[i] = (float) Math.sin(arg);
            sinEmb[(int)(i + half)] = (float) Math.cos(arg);
        }
        AccelTensor t = AccelTensor.fromFloatArray(sinEmb, 1, dim);
        
        // MLP
        AccelTensor w1 = weights.get("time_embedding.linear_1.weight");
        AccelTensor b1 = weights.get("time_embedding.linear_1.bias");
        AccelTensor w2 = weights.get("time_embedding.linear_2.weight");
        AccelTensor b2 = weights.get("time_embedding.linear_2.bias");
        
        AccelTensor h = AccelOps.add(AccelOps.linear(t, w1), b1);
        h = AccelOps.silu(h);
        return AccelOps.add(AccelOps.linear(h, w2), b2);
    }

    private AccelTensor resnetBlock(AccelTensor x, AccelTensor tEmbed, String base) {
        AccelTensor residual = x;
        
        // Part 1
        AccelTensor norm1 = AccelOps.groupNorm(x, weights.get(base + "norm1.weight"), weights.get(base + "norm1.bias"), 32, 1e-5);
        AccelTensor h = AccelOps.silu(norm1);
        h = AccelOps.conv2d(h, weights.get(base + "conv1.weight"), weights.get(base + "conv1.bias"), 1, 1);
        
        // Timestep addition
        AccelTensor tW = weights.get(base + "time_emb_proj.weight");
        AccelTensor tB = weights.get(base + "time_emb_proj.bias");
        AccelTensor tProj = AccelOps.add(AccelOps.linear(tEmbed, tW), tB); // [B, C]
        // Reshape tProj to [B, C, 1, 1] for broadcasting
        tProj = tProj.unsqueeze(-1).unsqueeze(-1);
        h = AccelOps.add(h, tProj);
        
        // Part 2
        AccelTensor norm2 = AccelOps.groupNorm(h, weights.get(base + "norm2.weight"), weights.get(base + "norm2.bias"), 32, 1e-5);
        h = AccelOps.silu(norm2);
        h = AccelOps.conv2d(h, weights.get(base + "conv2.weight"), weights.get(base + "conv2.bias"), 1, 1);
        
        // Shortcut
        if (weights.containsKey(base + "nin_shortcut.weight")) {
            residual = AccelOps.conv2d(residual, weights.get(base + "nin_shortcut.weight"), weights.get(base + "nin_shortcut.bias"), 1, 0);
        }
        
        return AccelOps.add(residual, h);
    }

    private AccelTensor spatialTransformer(AccelTensor x, AccelTensor context, String base) {
        long batch = x.size(0);
        long channels = x.size(1);
        long height = x.size(2);
        long width = x.size(3);
        long seqLen = height * width;

        AccelTensor residual = x;
        AccelTensor norm = AccelOps.groupNorm(x, weights.get(base + "norm.weight"), weights.get(base + "norm.bias"), 32, 1e-6);
        
        // Prepare for attention: [B, C, H, W] -> [B, H*W, C]
        AccelTensor h = norm.reshape(batch, channels, seqLen).transpose(1, 2);
        
        // Transformer Blocks (usually 1 in SD 1.4/1.5)
        String b = base + "transformer_blocks.0.";
        
        // 1. Self Attention
        AccelTensor r1 = h;
        AccelTensor n1 = AccelOps.layerNorm(h, weights.get(b + "norm1.weight"), weights.get(b + "norm1.bias"), 1e-6);
        h = AccelOps.add(r1, attention(n1, null, b + "attn1."));
        
        // 2. Cross Attention
        AccelTensor r2 = h;
        AccelTensor n2 = AccelOps.layerNorm(h, weights.get(b + "norm2.weight"), weights.get(b + "norm2.bias"), 1e-6);
        h = AccelOps.add(r2, attention(n2, context, b + "attn2."));
        
        // 3. MLP
        AccelTensor r3 = h;
        AccelTensor n3 = AccelOps.layerNorm(h, weights.get(b + "norm3.weight"), weights.get(b + "norm3.bias"), 1e-6);
        h = AccelOps.add(r3, mlp(n3, b + "ff."));
        
        // Back to image: [B, S, C] -> [B, C, H, W]
        AccelTensor out = h.transpose(1, 2).reshape(batch, channels, height, width);
        return AccelOps.add(residual, out);
    }

    private AccelTensor attention(AccelTensor x, AccelTensor context, String base) {
        boolean isCross = context != null;
        AccelTensor kv_input = isCross ? context : x;
        
        AccelTensor qW = weights.get(base + "to_q.weight");
        AccelTensor kW = weights.get(base + "to_k.weight");
        AccelTensor vW = weights.get(base + "to_v.weight");
        AccelTensor outW = weights.get(base + "to_out.0.weight");
        AccelTensor outB = weights.get(base + "to_out.0.bias");
        
        AccelTensor q = AccelOps.linear(x, qW);
        AccelTensor k = AccelOps.linear(kv_input, kW);
        AccelTensor v = AccelOps.linear(kv_input, vW);
        
        long batch = q.size(0);
        long seq = q.size(1);
        long embed = q.size(2);
        long heads = 8;
        long headDim = embed / heads;
        
        q = q.reshape(batch, seq, heads, headDim).transpose(1, 2);
        k = k.reshape(batch, k.size(1), heads, headDim).transpose(1, 2);
        v = v.reshape(batch, v.size(1), heads, headDim).transpose(1, 2);
        
        float scale = (float) (1.0 / Math.sqrt(headDim));
        AccelTensor scores = AccelOps.mulScalar(AccelOps.matmul(q, k.transpose(-2, -1)), scale);
        AccelTensor probs = AccelOps.softmax(scores, -1);
        AccelTensor attnOut = AccelOps.matmul(probs, v);
        
        AccelTensor merged = attnOut.transpose(1, 2).reshape(batch, seq, embed);
        return AccelOps.add(AccelOps.linear(merged, outW), outB);
    }

    private AccelTensor mlp(AccelTensor x, String base) {
        // base + "net.0.proj.weight" (GeGLU)
        AccelTensor w0 = weights.get(base + "net.0.proj.weight");
        AccelTensor w2 = weights.get(base + "net.2.weight");
        AccelTensor b2 = weights.get(base + "net.2.bias");

        AccelTensor h = AccelOps.linear(x, w0);
        // GeGLU: chunk h into gate and up, then silu(gate) * up
        long dim = h.size(-1) / 2;
        AccelTensor gate = h.slice(-1, 0, dim);
        AccelTensor up = h.slice(-1, dim, dim * 2);
        
        AccelTensor activated = AccelOps.mul(AccelOps.silu(gate), up);
        return AccelOps.add(AccelOps.linear(activated, w2), b2);
    }
}
