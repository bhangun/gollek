/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.rag;

/**
 * Interface for obtaining text embeddings.
 * Implementations may use local models, remote APIs, or cached embeddings.
 */
public interface EmbeddingProvider {

    /**
     * Generate embedding vector for the given text.
     *
     * @param text the input text
     * @return embedding vector
     */
    float[] embed(String text);

    /**
     * Generate embeddings for multiple texts in batch.
     *
     * @param texts the input texts
     * @return array of embedding vectors
     */
    default float[][] embedBatch(String[] texts) {
        float[][] results = new float[texts.length][];
        for (int i = 0; i < texts.length; i++) {
            results[i] = embed(texts[i]);
        }
        return results;
    }

    /**
     * Get the dimensionality of the embedding vectors.
     *
     * @return embedding dimension
     */
    int dimension();
}
