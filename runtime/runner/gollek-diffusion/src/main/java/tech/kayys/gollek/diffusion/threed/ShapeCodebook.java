package tech.kayys.gollek.diffusion.threed;

import java.util.Random;

/**
 * Discrete Vector-Quantized (VQ) shape codebook mapping discrete tokens {0, ..., V-1}
 * to continuous 3D geometric / latent feature vectors.
 */
public final class ShapeCodebook {

    private final int codebookSize; // V
    private final int embeddingDim;  // D
    private final float[][] embeddings;

    public ShapeCodebook(int codebookSize, int embeddingDim) {
        this.codebookSize = codebookSize;
        this.embeddingDim = embeddingDim;
        this.embeddings = new float[codebookSize][embeddingDim];
        initializeStructuredCodebook();
    }

    private void initializeStructuredCodebook() {
        Random rng = new Random(42);
        for (int i = 0; i < codebookSize; i++) {
            float norm = 0f;
            for (int d = 0; d < embeddingDim; d++) {
                float val = (float) (rng.nextGaussian() * 0.1);
                embeddings[i][d] = val;
                norm += val * val;
            }
            norm = (float) Math.sqrt(norm);
            if (norm > 1e-6f) {
                for (int d = 0; d < embeddingDim; d++) {
                    embeddings[i][d] /= norm;
                }
            }
        }
    }

    public int codebookSize() { return codebookSize; }
    public int embeddingDim() { return embeddingDim; }

    public float[] getEmbedding(int code) {
        if (code < 0 || code >= codebookSize) {
            return new float[embeddingDim];
        }
        return embeddings[code];
    }
}
