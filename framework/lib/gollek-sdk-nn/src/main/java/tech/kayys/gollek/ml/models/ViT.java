package tech.kayys.gollek.ml.models;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.TensorOps;
import tech.kayys.gollek.ml.nn.*;

/**
 * Vision Transformer (ViT) — image classification via pure self-attention.
 *
 * <p>Based on <em>"An Image is Worth 16x16 Words"</em> (Dosovitskiy et al., 2020).
 *
 * <p>Pipeline:
 * <pre>
 *   image [B,C,H,W] → patch embed [B,N,D] → prepend [CLS] → add pos enc
 *   → N × TransformerBlock → [CLS] token → MLP head → logits [B,numClasses]
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * NNModule vit = ViT.vitBase(numClasses = 1000);
 * GradTensor out = vit.forward(image); // [B,3,224,224] → [B,1000]
 * }</pre>
 */
public final class ViT {

    private ViT() {}

    /**
     * ViT-Base/16: patch=16, dModel=768, heads=12, layers=12, ~86M params.
     *
     * @param numClasses number of output classes
     * @return ViT-Base model
     */
    public static NNModule vitBase(int numClasses) {
        return new ViTModel(224, 16, 3, 768, 12, 12, 3072, numClasses, 0.1f);
    }

    /**
     * ViT-Small/16: patch=16, dModel=384, heads=6, layers=12, ~22M params.
     *
     * @param numClasses number of output classes
     * @return ViT-Small model
     */
    public static NNModule vitSmall(int numClasses) {
        return new ViTModel(224, 16, 3, 384, 6, 12, 1536, numClasses, 0.1f);
    }

    /**
     * ViT-Tiny/16: patch=16, dModel=192, heads=3, layers=12, ~5.7M params.
     *
     * @param numClasses number of output classes
     * @return ViT-Tiny model
     */
    public static NNModule vitTiny(int numClasses) {
        return new ViTModel(224, 16, 3, 192, 3, 12, 768, numClasses, 0.0f);
    }

    // ── Internal model ────────────────────────────────────────────────────

    static final class ViTModel extends NNModule {

        private final int patchSize, numPatches, dModel;
        private final Linear patchEmbed;   // [C*P*P, dModel]
        private final Parameter clsToken;  // [1, 1, dModel]
        private final PositionalEncoding posEnc;
        private final TransformerBlock[] blocks;
        private final LayerNorm norm;
        private final Linear head;

        /**
         * @param imgSize   input image size (square)
         * @param patchSize patch size (square)
         * @param inChannels input channels (3 for RGB)
         * @param dModel    embedding dimension
         * @param nHeads    attention heads
         * @param nLayers   transformer blocks
         * @param dFF       feed-forward hidden size
         * @param numClasses output classes
         * @param dropout   dropout rate
         */
        ViTModel(int imgSize, int patchSize, int inChannels, int dModel,
                 int nHeads, int nLayers, int dFF, int numClasses, float dropout) {
            this.patchSize  = patchSize;
            this.dModel     = dModel;
            int gridSize    = imgSize / patchSize;
            this.numPatches = gridSize * gridSize;

            this.patchEmbed = register("patch_embed",
                new Linear(inChannels * patchSize * patchSize, dModel));
            this.clsToken   = registerParameter("cls_token",
                GradTensor.zeros(1, 1, dModel));
            this.posEnc     = register("pos_enc",
                new PositionalEncoding(dModel, numPatches + 1));
            this.blocks     = new TransformerBlock[nLayers];
            for (int i = 0; i < nLayers; i++)
                blocks[i] = register("block_" + i, new TransformerBlock(dModel, nHeads, dFF, dropout));
            this.norm = register("norm", new LayerNorm(dModel));
            this.head = register("head", new Linear(dModel, numClasses));
        }

        /**
         * Forward pass: patch embed → prepend CLS → transformer → classify.
         *
         * @param x image tensor {@code [B, C, H, W]}
         * @return logits {@code [B, numClasses]}
         */
        @Override
        public GradTensor forward(GradTensor x) {
            long[] s = x.shape();
            int B = (int) s[0], C = (int) s[1], H = (int) s[2], W = (int) s[3];
            int P = patchSize, G = H / P;

            // Extract and flatten patches: [B, numPatches, C*P*P]
            float[] patches = extractPatches(x.data(), B, C, H, W, P, G);
            GradTensor tokens = patchEmbed.forward(
                GradTensor.of(patches, B, numPatches, C * P * P));

            // Prepend CLS token: [B, numPatches+1, dModel]
            GradTensor cls = GradTensor.of(clsToken.data().data().clone(), B, 1, dModel);
            tokens = TensorOps.cat(java.util.List.of(cls, tokens), 1);

            // Positional encoding + transformer blocks
            tokens = posEnc.forward(tokens);
            for (TransformerBlock block : blocks) tokens = block.forward(tokens);
            tokens = norm.forward(tokens);

            // Extract CLS token: [B, dModel]
            float[] clsOut = new float[B * dModel];
            for (int b = 0; b < B; b++)
                System.arraycopy(tokens.data(), b * (numPatches + 1) * dModel,
                                 clsOut, b * dModel, dModel);
            return head.forward(GradTensor.of(clsOut, B, dModel));
        }

        /** Extracts non-overlapping patches from an image tensor. */
        private float[] extractPatches(float[] img, int B, int C, int H, int W, int P, int G) {
            float[] out = new float[B * G * G * C * P * P];
            for (int b = 0; b < B; b++)
                for (int gh = 0; gh < G; gh++)
                    for (int gw = 0; gw < G; gw++) {
                        int patchIdx = b * G * G + gh * G + gw;
                        int outBase  = patchIdx * C * P * P;
                        for (int c = 0; c < C; c++)
                            for (int ph = 0; ph < P; ph++)
                                for (int pw = 0; pw < P; pw++) {
                                    int ih = gh * P + ph, iw = gw * P + pw;
                                    out[outBase + c * P * P + ph * P + pw] =
                                        img[b * C * H * W + c * H * W + ih * W + iw];
                                }
                    }
            return out;
        }
    }
}
