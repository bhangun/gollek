package tech.kayys.gollek.safetensor.runner.sd;

import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import java.util.Map;

/**
 * VAE Decoder implementation for Stable Diffusion.
 */
public class VAEDecoder {
    private final Map<String, AccelTensor> weights;

    public VAEDecoder(Map<String, AccelTensor> weights) {
        this.weights = weights;
    }

    public AccelTensor decode(AccelTensor z) {
        // Post-quantization scale
        AccelTensor x = AccelOps.mulScalar(z, 1.0f / 0.18215f);
        
        // Post-quant conv
        x = AccelOps.conv2d(x, weights.get("post_quant_conv.weight"), weights.get("post_quant_conv.bias"), 1, 0);
        
        // Initial convolution
        x = AccelOps.conv2d(x, weights.get("decoder.conv_in.weight"), weights.get("decoder.conv_in.bias"), 1, 1);
        
        // Mid Block
        x = resnetBlock(x, "decoder.mid.block_1.");
        x = attentionBlock(x, "decoder.mid.attn_1.");
        x = resnetBlock(x, "decoder.mid.block_2.");
        
        // Up Blocks (Simplified: loop through the 4 up blocks)
        for (int i = 0; i < 4; i++) {
            String base = "decoder.up_blocks." + i + ".";
            for (int j = 0; j < 3; j++) {
                x = resnetBlock(x, base + "resnets." + j + ".");
            }
            if (i < 3) {
                x = upsample(x, base + "upsamplers.0.");
            }
        }
        
        // Final normalization and output
        AccelTensor norm = AccelOps.groupNorm(x, weights.get("decoder.conv_norm_out.weight"), weights.get("decoder.conv_norm_out.bias"), 32, 1e-6);
        x = AccelOps.silu(norm);
        AccelTensor rgb = AccelOps.conv2d(x, weights.get("decoder.conv_out.weight"), weights.get("decoder.conv_out.bias"), 1, 1);
        
        // Map [-1, 1] to [0, 1]
        return AccelOps.addScalar(AccelOps.mulScalar(rgb, 0.5f), 0.5f);
    }

    private AccelTensor resnetBlock(AccelTensor x, String base) {
        AccelTensor residual = x;
        AccelTensor h = AccelOps.groupNorm(x, weights.get(base + "norm1.weight"), weights.get(base + "norm1.bias"), 32, 1e-6);
        h = AccelOps.silu(h);
        h = AccelOps.conv2d(h, weights.get(base + "conv1.weight"), weights.get(base + "conv1.bias"), 1, 1);
        
        h = AccelOps.groupNorm(h, weights.get(base + "norm2.weight"), weights.get(base + "norm2.bias"), 32, 1e-6);
        h = AccelOps.silu(h);
        h = AccelOps.conv2d(h, weights.get(base + "conv2.weight"), weights.get(base + "conv2.bias"), 1, 1);
        
        if (weights.containsKey(base + "nin_shortcut.weight")) {
            residual = AccelOps.conv2d(residual, weights.get(base + "nin_shortcut.weight"), weights.get(base + "nin_shortcut.bias"), 1, 0);
        }
        return AccelOps.add(residual, h);
    }

    private AccelTensor attentionBlock(AccelTensor x, String base) {
        long batch = x.size(0);
        long channels = x.size(1);
        long height = x.size(2);
        long width = x.size(3);
        
        AccelTensor residual = x;
        AccelTensor norm = AccelOps.groupNorm(x, weights.get(base + "group_norm.weight"), weights.get(base + "group_norm.bias"), 32, 1e-6);
        
        // Reshape for attention: [B, C, H, W] -> [B, H*W, C]
        AccelTensor h = norm.reshape(batch, channels, height * width).transpose(1, 2);
        
        AccelTensor q = AccelOps.linear(h, weights.get(base + "q.weight"));
        AccelTensor k = AccelOps.linear(h, weights.get(base + "k.weight"));
        AccelTensor v = AccelOps.linear(h, weights.get(base + "v.weight"));
        
        // Single head attention
        float scale = (float) (1.0 / Math.sqrt(channels));
        AccelTensor scores = AccelOps.mulScalar(AccelOps.matmul(q, k.transpose(-2, -1)), scale);
        AccelTensor probs = AccelOps.softmax(scores, -1);
        AccelTensor attnOut = AccelOps.matmul(probs, v);
        
        AccelTensor out = AccelOps.linear(attnOut, weights.get(base + "proj_out.weight"));
        out = out.transpose(1, 2).reshape(batch, channels, height, width);
        
        return AccelOps.add(residual, out);
    }

    private AccelTensor upsample(AccelTensor x, String base) {
        // Nearest neighbor upsample 2x
        long b = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
        AccelTensor up = AccelTensor.zeros(b, c, h * 2, w * 2);
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
}
