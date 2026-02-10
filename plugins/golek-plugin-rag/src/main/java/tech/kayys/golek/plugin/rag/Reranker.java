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
 * Interface for reranking retrieved documents.
 * Reranking improves retrieval quality by using a more sophisticated scoring method
 * than the initial vector similarity search.
 */
public interface Reranker {

    /**
     * Rerank retrieved documents based on relevance to the query.
     *
     * @param query     the original query text
     * @param documents the retrieved documents to rerank
     * @param topK      maximum number of results after reranking
     * @return reranked list of documents (best first)
     */
    List<RetrievalService.RetrievedDocument> rerank(
            String query,
            List<RetrievalService.RetrievedDocument> documents,
            int topK);
}
