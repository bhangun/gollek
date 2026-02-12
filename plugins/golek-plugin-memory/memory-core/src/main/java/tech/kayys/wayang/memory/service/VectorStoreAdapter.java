package tech.kayys.wayang.memory.service;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.memory.model.Memory;
import tech.kayys.wayang.memory.model.ScoredMemory;
import tech.kayys.wayang.vector.VectorEntry;
import tech.kayys.wayang.vector.VectorQuery;
import tech.kayys.wayang.vector.VectorStore;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Adapter to bridge the VectorStore interface from the vector module
 * with the VectorMemoryStore interface from the memory module.
 */
public class VectorStoreAdapter implements VectorMemoryStore {

    private final VectorStore vectorStore;

    public VectorStoreAdapter(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    @Override
    public Uni<String> store(Memory memory) {
        VectorEntry vectorEntry = new VectorEntry(
            memory.getId(),
            List.of(toFloatArray(memory.getEmbedding())),
            memory.getContent(),
            memory.getMetadata()
        );

        return vectorStore.store(List.of(vectorEntry))
            .onItem().transform(v -> memory.getId());
    }

    @Override
    public Uni<List<String>> storeBatch(List<Memory> memories) {
        List<VectorEntry> vectorEntries = memories.stream()
            .map(this::toVectorEntry)
            .collect(Collectors.toList());

        return vectorStore.store(vectorEntries)
            .onItem().transform(v -> 
                memories.stream().map(Memory::getId).collect(Collectors.toList())
            );
    }

    @Override
    public Uni<List<ScoredMemory>> search(float[] queryEmbedding, int limit, double minSimilarity, Map<String, Object> filters) {
        VectorQuery query = new VectorQuery(List.of(toFloatArray(queryEmbedding)), limit, (float) minSimilarity);

        return vectorStore.search(query, filters)
            .map(entries -> entries.stream()
                .map(this::toScoredMemory)
                .collect(Collectors.toList()));
    }

    @Override
    public Uni<List<ScoredMemory>> hybridSearch(float[] queryEmbedding, List<String> keywords, int limit, double semanticWeight) {
        // For now, delegate to regular search - a full implementation would combine semantic and keyword search
        return search(queryEmbedding, limit, 0.0, null);
    }

    @Override
    public Uni<Memory> retrieve(String memoryId) {
        // For direct retrieval, we'll search with a filter for the specific ID
        Map<String, Object> filters = Map.of("id", memoryId);
        VectorQuery query = new VectorQuery(List.of(), 1, 0.0f);

        return vectorStore.search(query, filters)
            .map(entries -> entries.isEmpty() ? null : toMemory(entries.get(0)));
    }

    @Override
    public Uni<List<Memory>> retrieveBatch(List<String> memoryIds) {
        // For batch retrieval, we'll search with a filter for the specific IDs
        Map<String, Object> filters = Map.of("ids", memoryIds);
        VectorQuery query = new VectorQuery(List.of(), memoryIds.size(), 0.0f);

        return vectorStore.search(query, filters)
            .map(entries -> entries.stream()
                .map(this::toMemory)
                .collect(Collectors.toList()));
    }

    @Override
    public Uni<Memory> updateMetadata(String memoryId, Map<String, Object> metadata) {
        // This would require a more sophisticated approach in a real implementation
        // For now, we'll retrieve the memory, update its metadata, and store it again
        return retrieve(memoryId)
            .flatMap(memory -> {
                if (memory == null) {
                    return Uni.createFrom().nullItem();
                }

                // Update metadata
                Map<String, Object> updatedMetadata = memory.getMetadata();
                updatedMetadata.putAll(metadata);

                // Create updated memory
                Memory updatedMemory = Memory.builder()
                    .id(memory.getId())
                    .namespace(memory.getNamespace())
                    .content(memory.getContent())
                    .embedding(memory.getEmbedding())
                    .type(memory.getType())
                    .metadata(updatedMetadata)
                    .timestamp(memory.getTimestamp())
                    .expiresAt(memory.getExpiresAt())
                    .importance(memory.getImportance())
                    .build();

                return store(updatedMemory).map(id -> updatedMemory);
            });
    }

    @Override
    public Uni<Boolean> delete(String memoryId) {
        return vectorStore.delete(List.of(memoryId))
            .onItem().transform(v -> true);
    }

    @Override
    public Uni<Long> deleteNamespace(String namespace) {
        Map<String, Object> filters = Map.of("namespace", namespace);
        // For now, we'll just return 1 as a placeholder - a full implementation would count actual deletions
        return vectorStore.deleteByFilters(filters)
            .onItem().transform(v -> 1L);
    }

    @Override
    public Uni<MemoryStatistics> getStatistics(String namespace) {
        // This would require specific implementation depending on the underlying store
        // For now, return a default statistics object
        return Uni.createFrom().item(new MemoryStatistics(
            namespace, 0, 0, 0, 0, 0, 0.0, null, null
        ));
    }

    /**
     * Convert Memory to VectorEntry
     */
    private VectorEntry toVectorEntry(Memory memory) {
        return new VectorEntry(
            memory.getId(),
            List.of(toFloatArray(memory.getEmbedding())),
            memory.getContent(),
            memory.getMetadata()
        );
    }

    /**
     * Convert VectorEntry to Memory
     */
    private Memory toMemory(VectorEntry vectorEntry) {
        return Memory.builder()
            .id(vectorEntry.id())
            .content(vectorEntry.content())
            .embedding(toFloatArray(vectorEntry.vector().toArray(new Float[0])))
            .metadata(vectorEntry.metadata())
            .build();
    }

    /**
     * Convert VectorEntry to ScoredMemory (with a default score)
     */
    private ScoredMemory toScoredMemory(VectorEntry vectorEntry) {
        Memory memory = toMemory(vectorEntry);
        // For now, we'll use a default score of 1.0 since VectorEntry doesn't have score info
        return new ScoredMemory(memory, 1.0);
    }

    /**
     * Convert float array to Float list
     */
    private Float[] toFloatArray(float[] array) {
        if (array == null) return new Float[0];
        Float[] result = new Float[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }

    /**
     * Convert Float list to float array
     */
    private float[] toFloatArray(Float[] array) {
        if (array == null) return new float[0];
        float[] result = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }
}
