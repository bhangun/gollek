package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * A lookup table that maps integer token IDs to dense embedding vectors.
 * <p>
 * Embeddings are the foundation of NLP and sequence models. Each vocabulary token
 * is assigned a learnable dense vector representation. During forward pass, token IDs
 * are converted to their corresponding embeddings.
 * <p>
 * The embedding matrix is initialized with a normal distribution (mean=0, std=0.02)
 * which is common practice in transformer models.
 * <p>
 * Equivalent to {@code torch.nn.Embedding}.
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Input:</b> arbitrary shape containing integer token IDs (as floats)</li>
 *   <li><b>Embedding Table:</b> [numEmbeddings, embeddingDim]</li>
 *   <li><b>Output:</b> input shape + [embeddingDim] at the end</li>
 * </ul>
 *
 * <h3>Examples</h3>
 * <pre>{@code
 * // Create embedding layer: vocab size 50K, embedding dim 768
 * var embed = new Embedding(50000, 768);
 *
 * // Input: token IDs [batch, seq_len]
 * var tokenIds = GradTensor.of(new float[]{1, 45, 200, ...}, 2, 3);
 *
 * // Output: embeddings [batch, seq_len, 768]
 * var embeddings = embed.forward(tokenIds);  // shape: [2, 3, 768]
 * }</pre>
 *
 * <h3>Typical Usage in Transformer</h3>
 * <pre>{@code
 * var embedding = new Embedding(50257, 768);  // GPT-2 vocab
 * var posEncoding = new PositionalEncoding(768);
 * var tokens = GradTensor.of(new float[]{...}, 1, 10);  // [batch=1, seq=10]
 * var x = embedding.forward(tokens);  // [1, 10, 768]
 * x = posEncoding.forward(x);
 * }</pre>
 *
 * <h3>Common Vocabulary Sizes</h3>
 * <ul>
 *   <li>GPT-2: 50,257 tokens</li>
 *   <li>GPT-3: 50,257 tokens</li>
 *   <li>BERT: 30,522 tokens</li>
 *   <li>Custom models: typically 10K-100K</li>
 * </ul>
 */
public class Embedding extends NNModule {

    private final int numEmbeddings;
    private final int embeddingDim;
    private final Parameter weight;

    /**
     * Create an embedding layer.
     *
     * @param numEmbeddings size of the vocabulary (number of unique tokens)
     * @param embeddingDim  dimension of each embedding vector
     *
     * @throws IllegalArgumentException if numEmbeddings or embeddingDim is non-positive
     */
    public Embedding(int numEmbeddings, int embeddingDim) {
        if (numEmbeddings <= 0) {
            throw new IllegalArgumentException("numEmbeddings must be positive, got: " + numEmbeddings);
        }
        if (embeddingDim <= 0) {
            throw new IllegalArgumentException("embeddingDim must be positive, got: " + embeddingDim);
        }

        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;

        // Initialize with normal distribution N(0, 0.02)
        float[] data = new float[numEmbeddings * embeddingDim];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) rng.nextGaussian() * 0.02f;
        }
        this.weight = registerParameter("weight", GradTensor.of(data, numEmbeddings, embeddingDim));
    }

    /**
     * Look up embeddings for the given token IDs.
     *
     * @param input tensor containing integer token IDs as floats, arbitrary shape
     * @return embeddings tensor with shape: input.shape() + [embeddingDim]
     *
     * @throws IndexOutOfBoundsException if any token ID is out of range [0, numEmbeddings)
     */
    @Override
    public GradTensor forward(GradTensor input) {
        float[] ids = input.data();
        long[] inShape = input.shape();
        int seqLen = ids.length;
        float[] result = new float[seqLen * embeddingDim];
        float[] wData = weight.data().data();

        // Look up each token ID in the embedding table
        for (int i = 0; i < seqLen; i++) {
            int tokenId = (int) ids[i];
            if (tokenId < 0 || tokenId >= numEmbeddings) {
                throw new IndexOutOfBoundsException(
                    "Token ID " + tokenId + " out of valid range [0, " + (numEmbeddings - 1) + "]");
            }
            // Copy the embedding vector for this token
            System.arraycopy(wData, tokenId * embeddingDim, result, i * embeddingDim, embeddingDim);
        }

        // Output shape: [...original input shape..., embeddingDim]
        long[] outShape = new long[inShape.length + 1];
        System.arraycopy(inShape, 0, outShape, 0, inShape.length);
        outShape[inShape.length] = embeddingDim;

        GradTensor out = GradTensor.of(result, outShape);
        if (weight.data().requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("Embedding") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] ug = upstream.data();
                    float[] wGrad = new float[numEmbeddings * embeddingDim];
                    // Accumulate gradients: each token's gradient adds to its embedding vector
                    for (int i = 0; i < seqLen; i++) {
                        int tokenId = (int) ids[i];
                        for (int j = 0; j < embeddingDim; j++) {
                            wGrad[tokenId * embeddingDim + j] += ug[i * embeddingDim + j];
                        }
                    }
                    weight.data().backward(GradTensor.of(wGrad, weight.data().shape()));
                }
            });
        }
        return out;
    }

    /**
     * Get the vocabulary size.
     *
     * @return numEmbeddings
     */
    public int getNumEmbeddings() {
        return numEmbeddings;
    }

    /**
     * Get the embedding vector dimension.
     *
     * @return embeddingDim
     */
    public int getEmbeddingDim() {
        return embeddingDim;
    }

    @Override
    public String toString() {
        return "Embedding(num=" + numEmbeddings + ", dim=" + embeddingDim + ")";
    }
}
