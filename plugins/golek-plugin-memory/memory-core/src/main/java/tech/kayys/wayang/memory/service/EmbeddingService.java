package tech.kayys.gamelan.executor.memory;

import io.smallrye.mutiny.Uni;
import java.util.List;

/**
 * Embedding service interface
 */
public interface EmbeddingService {

    /**
     * Generate embedding for a single text
     *
     * @param text Text to embed
     * @return Embedding vector
     */
    Uni<float[]> embed(String text);

    /**
     * Generate embeddings for multiple texts in batch
     *
     * @param texts List of texts to embed
     * @return List of embedding vectors
     */
    Uni<List<float[]>> embedBatch(List<String> texts);

    /**
     * Get the dimension of embeddings produced by this service
     *
     * @return Embedding dimension
     */
    int getDimension();

    /**
     * Get the provider name
     *
     * @return Provider name (e.g., "openai", "cohere", "local")
     */
    String getProvider();
}