package tech.kayys.gamelan.executor.memory;

import io.smallrye.mutiny.Uni;

import java.util.List;
import java.util.Map;

/**
 * Interface for storing and retrieving vector-based memories
 */
public interface VectorMemoryStore {
        /**
         * Store a single memory
         */
        Uni<String> store(Memory memory);

        /**
         * Store multiple memories in batch
         */
        Uni<List<String>> storeBatch(List<Memory> memories);

        /**
         * Search for similar memories using vector similarity
         */
        Uni<List<ScoredMemory>> search(
                        float[] queryEmbedding,
                        int limit,
                        double minSimilarity,
                        Map<String, Object> filters);

        /**
         * Perform hybrid search combining semantic and keyword search
         */
        Uni<List<ScoredMemory>> hybridSearch(
                        float[] queryEmbedding,
                        List<String> keywords,
                        int limit,
                        double semanticWeight);

        /**
         * Retrieve a single memory by ID
         */
        Uni<Memory> retrieve(String memoryId);

        /**
         * Retrieve multiple memories by IDs
         */
        Uni<List<Memory>> retrieveBatch(List<String> memoryIds);

        /**
         * Update metadata for a memory
         */
        Uni<Memory> updateMetadata(String memoryId, Map<String, Object> metadata);

        /**
         * Delete a memory by ID
         */
        Uni<Boolean> delete(String memoryId);

        /**
         * Delete all memories in a namespace
         */
        Uni<Long> deleteNamespace(String namespace);

        /**
         * Get statistics about memories in a namespace
         */
        Uni<MemoryStatistics> getStatistics(String namespace);
}