/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.rag;

import java.util.List;

/**
 * Interface for vector search and document retrieval.
 * Implementations may use vector databases (Qdrant, Pinecone, Weaviate, etc.)
 * or in-memory indices.
 */
public interface RetrievalService {

    /**
     * Retrieve documents similar to the given embedding vector.
     *
     * @param embedding           the query embedding vector
     * @param topK                maximum number of results
     * @param similarityThreshold minimum similarity score (0.0 - 1.0)
     * @return list of retrieved documents sorted by similarity
     */
    List<RetrievedDocument> retrieve(float[] embedding, int topK, double similarityThreshold);

    /**
     * Retrieve documents matching the given text query.
     * Uses full-text search or hybrid (vector + text) search.
     *
     * @param query the text query
     * @param topK  maximum number of results
     * @return list of retrieved documents
     */
    default List<RetrievedDocument> retrieveByText(String query, int topK) {
        return List.of(); // Default: not supported
    }

    /**
     * A retrieved document with content and metadata.
     *
     * @param content    the document content
     * @param source     the source identifier (e.g., file path, URL)
     * @param score      similarity score (0.0 - 1.0)
     * @param chunkIndex the chunk index within the source document
     */
    record RetrievedDocument(
            String content,
            String source,
            double score,
            int chunkIndex) {
    }
}
