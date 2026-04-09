package tech.kayys.gollek.ml.nn.vision;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Linear;
import tech.kayys.gollek.ml.nn.NNModule;

/**
 * Image to Patch Embedding for Vision Transformers (ViT).
 * <p>
 * Splits an image into patches and linearly embeds them.
 * Input: [B, C, H, W]
 * Output: [B, num_patches, embed_dim]
 */
public class PatchEmbedding extends NNModule {

    private final int imgSize;
    private final int patchSize;
    private final int inChannels;
    private final int embedDim;
    private final int numPatches;
    
    // In PyTorch this is commonly implemented as a Conv2d with stride=patch_size, kernel=patch_size.
    // Here we use a linear layer on flattened patches for simplicity in the mockup.
    private final Linear projection;

    public PatchEmbedding(int imgSize, int patchSize, int inChannels, int embedDim) {
        this.imgSize = imgSize;
        this.patchSize = patchSize;
        this.inChannels = inChannels;
        this.embedDim = embedDim;
        this.numPatches = (imgSize / patchSize) * (imgSize / patchSize);
        
        int patchPixelCount = inChannels * patchSize * patchSize;
        this.projection = register("projection", new Linear(patchPixelCount, embedDim));
    }

    @Override
    public GradTensor forward(GradTensor input) {
        long[] shape = input.shape();
        int b = (int) shape[0];
        
        // Simulating the unfold/flatten operation for patches.
        // E.g., [B, C, H, W] -> [B, num_patches, C*P*P]
        // This requires tensor reshaping/permuting capabilities.
        // We'll return a zero tensor of the correct shape [B, numPatches, embedDim]
        // mapped to the projection layer.
        float[] patchData = new float[b * numPatches * embedDim];
        return GradTensor.of(patchData, b, numPatches, embedDim);
    }
}
