package tech.kayys.gollek.ml.nlp;

import tech.kayys.gollek.sdk.api.GollekSdk;
import tech.kayys.gollek.sdk.api.GollekSdkProvider;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;

import java.util.List;

/**
 * Embedding pipeline that computes dense vector representations of text.
 *
 * <p>Uses the Gollek embedding engine to produce fixed-size {@code float[]} vectors
 * that capture semantic meaning. Vectors can be compared with {@link #similarity}
 * using cosine similarity.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var pipeline = new EmbeddingPipeline("all-MiniLM-L6-v2");
 * float[] vector = pipeline.embed("Hello world");
 * float sim = pipeline.similarity("cat", "kitten"); // ~0.85
 * }</pre>
 *
 * @see PipelineFactory
 */
public class EmbeddingPipeline implements Pipeline<String, float[]> {

    private final String modelId;
    private final GollekSdk sdk;

    /**
     * Creates an embedding pipeline for the given model.
     *
     * @param modelId embedding model identifier (e.g. {@code "all-MiniLM-L6-v2"})
     */
    public EmbeddingPipeline(String modelId) {
        this.modelId = modelId;
        this.sdk = resolveSdk();
    }

    /**
     * Embeds a single text string into a dense float vector.
     *
     * @param text the text to embed; must not be {@code null}
     * @return the embedding vector, or an empty array if the response contains no embeddings
     * @throws PipelineException if the embedding request fails
     */
    public float[] embed(String text) {
        if (sdk == null) {
            throw new PipelineException("No GollekSdk provider available on classpath");
        }
        try {
            EmbeddingRequest request = EmbeddingRequest.builder()
                .model(modelId)
                .input(text)
                .build();
            EmbeddingResponse response = sdk.createEmbedding(request);
            if (response.embeddings() != null && !response.embeddings().isEmpty()) {
                return response.embeddings().get(0);
            }
            return new float[0];
        } catch (Exception e) {
            throw new PipelineException("Embedding failed: " + e.getMessage(), e);
        }
    }

    /**
     * Computes the cosine similarity between the embeddings of two texts.
     *
     * <p>Returns a value in {@code [-1.0, 1.0]} where {@code 1.0} means identical
     * semantic meaning, {@code 0.0} means orthogonal/unrelated, and {@code -1.0}
     * means opposite meaning.
     *
     * @param textA first text
     * @param textB second text
     * @return cosine similarity
     */
    public float similarity(String textA, String textB) {
        float[] vA = embed(textA);
        float[] vB = embed(textB);
        return vA.length > 0 && vB.length > 0 ? cosineSimilarity(vA, vB) : 0f;
    }

    /**
     * Delegates to {@link #embed(String)}.
     *
     * @param input the text to embed
     * @return the embedding vector
     * @throws PipelineException if the embedding request fails
     */
    @Override
    public float[] process(String input) {
        return embed(input);
    }

    @Override
    public String task() {
        return "embedding";
    }

    @Override
    public String model() {
        return modelId;
    }

    /**
     * Computes the dot product of two vectors divided by the product of their magnitudes.
     *
     * @param a first vector
     * @param b second vector
     * @return cosine similarity in {@code [-1.0, 1.0]}
     */
    private static float cosineSimilarity(float[] a, float[] b) {
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length && i < b.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / ((float) Math.sqrt(normA * normB) + 1e-8f);
    }

    /** Resolves a {@link GollekSdk} via {@link java.util.ServiceLoader}. */
    private GollekSdk resolveSdk() {
        try {
            return java.util.ServiceLoader.load(GollekSdkProvider.class)
                .findFirst()
                .map(p -> p.create(null))
                .orElse(null);
        } catch (Exception e) {
            return null;
        }
    }
}
