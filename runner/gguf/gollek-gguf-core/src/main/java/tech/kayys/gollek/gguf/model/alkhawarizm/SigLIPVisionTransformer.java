package tech.kayys.gollek.gguf.model.alkhawarizm;

import tech.kayys.alkhawarizm.core.nn.Conv2d;
import tech.kayys.alkhawarizm.core.nn.LayerNorm;
import tech.kayys.alkhawarizm.core.nn.Linear;
import tech.kayys.alkhawarizm.core.nn.Module;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.tensor.WeightAdapter;
import tech.kayys.gollek.gguf.model.ModelConfig;

import java.util.ArrayList;
import java.util.List;

public class SigLIPVisionTransformer extends Module {

    public static class PatchEmbedding extends Module {
        private final Conv2d proj;
        private final Tensor positionEmbedding;

        public PatchEmbedding(WeightAdapter weights, int embedDim, int patchSize) {
            // Note: WeightAdapter needs to map "vision.patch_embedding.weight" to Conv2d
            // weights
            Tensor wProj = weights.getWeight("vpm.patch_embd.proj.weight");
            Tensor bProj = weights.getWeight("vpm.patch_embd.proj.bias");

            proj = new Conv2d(wProj, bProj, patchSize, patchSize, patchSize, patchSize);
            registerModule("proj", proj);

            positionEmbedding = weights.getWeight("vpm.patch_embd.position_embedding.weight");
            registerParameter("position_embedding", positionEmbedding);
        }

        @Override
        public Tensor forward(Tensor pixelValues) {
            // pixelValues: [batch, channels, height, width]
            Tensor patchEmbeds = proj.forward(pixelValues);

            // Flatten spatial dimensions and transpose
            // from [batch, embedDim, gridH, gridW] to [batch, gridH * gridW, embedDim]
            long batch = patchEmbeds.shape().dim(0);
            long embedDim = patchEmbeds.shape().dim(1);
            long gridH = patchEmbeds.shape().dim(2);
            long gridW = patchEmbeds.shape().dim(3);

            patchEmbeds = patchEmbeds.reshape(batch, embedDim, gridH * gridW).transpose(1, 2);

            // Add position embeddings
            return patchEmbeds.add(positionEmbedding);
        }
    }

    public static class SigLIPAttention extends Module {
        private final Linear qProj, kProj, vProj, outProj;
        private final int nHeads;
        private final int headDim;

        public SigLIPAttention(WeightAdapter weights, int layerIdx, int embedDim, int nHeads) {
            this.nHeads = nHeads;
            this.headDim = embedDim / nHeads;

            String prefix = "vpm.blk." + layerIdx + ".attn_";
            qProj = new Linear(weights.getWeight(prefix + "q.weight"), weights.getWeight(prefix + "q.bias"));
            kProj = new Linear(weights.getWeight(prefix + "k.weight"), weights.getWeight(prefix + "k.bias"));
            vProj = new Linear(weights.getWeight(prefix + "v.weight"), weights.getWeight(prefix + "v.bias"));
            outProj = new Linear(weights.getWeight(prefix + "out.weight"), weights.getWeight(prefix + "out.bias"));

            registerModule("q_proj", qProj);
            registerModule("k_proj", kProj);
            registerModule("v_proj", vProj);
            registerModule("out_proj", outProj);
        }

        @Override
        public Tensor forward(Tensor x) {
            long batch = x.shape().dim(0);
            long seqLen = x.shape().dim(1);

            Tensor q = qProj.forward(x).reshape(batch, seqLen, nHeads, headDim).transpose(1, 2);
            Tensor k = kProj.forward(x).reshape(batch, seqLen, nHeads, headDim).transpose(1, 2);
            Tensor v = vProj.forward(x).reshape(batch, seqLen, nHeads, headDim).transpose(1, 2);

            Tensor scores = q.matmul(k.transpose(-2, -1)).div((float) Math.sqrt(headDim));
            Tensor probs = scores.softmax(-1);
            Tensor out = probs.matmul(v).transpose(1, 2).reshape(batch, seqLen, (long) nHeads * headDim);

            return outProj.forward(out);
        }
    }

    public static class SigLIPMLP extends Module {
        private final Linear fc1, fc2;

        public SigLIPMLP(WeightAdapter weights, int layerIdx) {
            String prefix = "vpm.blk." + layerIdx + ".ffn_";
            // SigLIP uses fc1 and fc2 for MLP, or gate/up/down depending on variant
            fc1 = new Linear(weights.getWeight(prefix + "fc1.weight"), weights.getWeight(prefix + "fc1.bias"));
            fc2 = new Linear(weights.getWeight(prefix + "fc2.weight"), weights.getWeight(prefix + "fc2.bias"));

            registerModule("fc1", fc1);
            registerModule("fc2", fc2);
        }

        @Override
        public Tensor forward(Tensor x) {
            return fc2.forward(fc1.forward(x).gelu()); // GELU activation
        }
    }

    public static class SigLIPEncoderLayer extends Module {
        private final LayerNorm norm1, norm2;
        private final SigLIPAttention attn;
        private final SigLIPMLP mlp;

        public SigLIPEncoderLayer(WeightAdapter weights, int layerIdx, int embedDim, int nHeads, float eps) {
            String prefix = "vpm.blk." + layerIdx + ".";
            norm1 = new LayerNorm(weights.getWeight(prefix + "attn_norm.weight"),
                    weights.getWeight(prefix + "attn_norm.bias"), eps);
            attn = new SigLIPAttention(weights, layerIdx, embedDim, nHeads);

            norm2 = new LayerNorm(weights.getWeight(prefix + "ffn_norm.weight"),
                    weights.getWeight(prefix + "ffn_norm.bias"), eps);
            mlp = new SigLIPMLP(weights, layerIdx);

            registerModule("norm1", norm1);
            registerModule("attn", attn);
            registerModule("norm2", norm2);
            registerModule("mlp", mlp);
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor residual = x;
            x = attn.forward(norm1.forward(x)).add(residual);
            residual = x;
            x = mlp.forward(norm2.forward(x)).add(residual);
            return x;
        }
    }

    private final PatchEmbedding patchEmbed;
    private final List<SigLIPEncoderLayer> layers = new ArrayList<>();
    private final LayerNorm postLayerNorm;

    public SigLIPVisionTransformer(ModelConfig cfg, WeightAdapter weights) {
        // PaliGemma defaults (these could be dynamically loaded from cfg)
        int embedDim = 1152;
        int patchSize = 14;
        int nHeads = 16;
        int nLayers = 27;
        float eps = 1e-6f;

        patchEmbed = new PatchEmbedding(weights, embedDim, patchSize);
        registerModule("patch_embed", patchEmbed);

        for (int i = 0; i < nLayers; i++) {
            SigLIPEncoderLayer layer = new SigLIPEncoderLayer(weights, i, embedDim, nHeads, eps);
            layers.add(layer);
            registerModule("layer_" + i, layer);
        }

        postLayerNorm = new LayerNorm(weights.getWeight("vpm.post_norm.weight"),
                weights.getWeight("vpm.post_norm.bias"), eps);
        registerModule("post_norm", postLayerNorm);
    }

    @Override
    public Tensor forward(Tensor pixelValues) {
        Tensor hiddenStates = patchEmbed.forward(pixelValues);
        for (SigLIPEncoderLayer layer : layers) {
            hiddenStates = layer.forward(hiddenStates);
        }
        return postLayerNorm.forward(hiddenStates);
    }
}
