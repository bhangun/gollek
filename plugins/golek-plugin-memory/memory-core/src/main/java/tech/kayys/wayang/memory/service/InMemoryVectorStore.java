package tech.kayys.gamelan.executor.memory;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * In-memory implementation of VectorMemoryStore for development and testing
 */
@ApplicationScoped
public class InMemoryVectorStore implements VectorMemoryStore {
    private static final Logger LOG = LoggerFactory.getLogger(InMemoryVectorStore.class);

    // Storage: memoryId -> Memory
    private final Map<String, Memory> memoryStore = new ConcurrentHashMap<>();

    @Override
    public Uni<String> store(Memory memory) {
        LOG.debug("Storing memory: {}", memory.getId());

        memoryStore.put(memory.getId(), memory);

        return Uni.createFrom().item(memory.getId());
    }

    @Override
    public Uni<List<String>> storeBatch(List<Memory> memories) {
        LOG.debug("Storing batch of {} memories", memories.size());

        List<String> ids = new ArrayList<>();
        for (Memory memory : memories) {
            memoryStore.put(memory.getId(), memory);
            ids.add(memory.getId());
        }

        return Uni.createFrom().item(ids);
    }

    @Override
    public Uni<List<ScoredMemory>> search(
            float[] queryEmbedding,
            int limit,
            double minSimilarity,
            Map<String, Object> filters) {

        LOG.debug("Searching for similar memories with limit: {}", limit);

        List<ScoredMemory> results = new ArrayList<>();

        for (Memory memory : memoryStore.values()) {
            // Apply filters
            if (!matchesFilters(memory, filters)) {
                continue;
            }

            // Calculate cosine similarity
            double similarity = cosineSimilarity(queryEmbedding, memory.getEmbedding());

            if (similarity >= minSimilarity) {
                results.add(new ScoredMemory(memory, similarity));
            }
        }

        // Sort by similarity (descending)
        results.sort((a, b) -> Double.compare(b.getScore(), a.getScore()));

        // Limit results
        if (results.size() > limit) {
            results = results.subList(0, limit);
        }

        LOG.debug("Found {} similar memories", results.size());

        return Uni.createFrom().item(results);
    }

    @Override
    public Uni<List<ScoredMemory>> hybridSearch(
            float[] queryEmbedding,
            List<String> keywords,
            int limit,
            double semanticWeight) {

        LOG.debug("Performing hybrid search with {} keywords", keywords.size());

        // For in-memory implementation, we'll just do semantic search
        // A real implementation would combine keyword matching with semantic search
        return search(queryEmbedding, limit, 0.0, new HashMap<>());
    }

    @Override
    public Uni<Memory> retrieve(String memoryId) {
        LOG.debug("Retrieving memory: {}", memoryId);

        Memory memory = memoryStore.get(memoryId);
        return Uni.createFrom().item(memory);
    }

    @Override
    public Uni<List<Memory>> retrieveBatch(List<String> memoryIds) {
        LOG.debug("Retrieving batch of {} memories", memoryIds.size());

        List<Memory> memories = new ArrayList<>();
        for (String id : memoryIds) {
            Memory memory = memoryStore.get(id);
            if (memory != null) {
                memories.add(memory);
            }
        }

        return Uni.createFrom().item(memories);
    }

    @Override
    public Uni<Memory> updateMetadata(String memoryId, Map<String, Object> metadata) {
        LOG.debug("Updating metadata for memory: {}", memoryId);

        Memory existing = memoryStore.get(memoryId);
        if (existing == null) {
            return Uni.createFrom().nullItem();
        }

        // Create updated memory with new metadata
        Memory updated = Memory.builder()
                .id(existing.getId())
                .namespace(existing.getNamespace())
                .content(existing.getContent())
                .embedding(existing.getEmbedding())
                .type(existing.getType())
                .metadata(mergeMetadata(existing.getMetadata(), metadata))
                .timestamp(existing.getTimestamp())
                .expiresAt(existing.getExpiresAt())
                .importance(existing.getImportance())
                .build();

        memoryStore.put(memoryId, updated);

        return Uni.createFrom().item(updated);
    }

    @Override
    public Uni<Boolean> delete(String memoryId) {
        LOG.debug("Deleting memory: {}", memoryId);

        boolean deleted = memoryStore.remove(memoryId) != null;
        return Uni.createFrom().item(deleted);
    }

    @Override
    public Uni<Long> deleteNamespace(String namespace) {
        LOG.debug("Deleting all memories in namespace: {}", namespace);

        long count = memoryStore.values().removeIf(memory -> namespace.equals(memory.getNamespace()));

        return Uni.createFrom().item(count);
    }

    @Override
    public Uni<MemoryStatistics> getStatistics(String namespace) {
        LOG.debug("Getting statistics for namespace: {}", namespace);

        List<Memory> memories = memoryStore.values().stream()
                .filter(memory -> namespace.equals(memory.getNamespace()))
                .collect(Collectors.toList());

        long total = memories.size();
        long episodic = memories.stream()
                .filter(m -> m.getType() == tech.kayys.gamelan.core.domain.MemoryType.EPISODIC)
                .count();
        long semantic = memories.stream()
                .filter(m -> m.getType() == tech.kayys.gamelan.core.domain.MemoryType.SEMANTIC)
                .count();
        long procedural = memories.stream()
                .filter(m -> m.getType() == tech.kayys.gamelan.core.domain.MemoryType.PROCEDURAL)
                .count();
        long working = memories.stream()
                .filter(m -> m.getType() == tech.kayys.gamelan.core.domain.MemoryType.WORKING)
                .count();

        double avgImportance = memories.stream()
                .mapToDouble(Memory::getImportance)
                .average()
                .orElse(0.0);

        Instant oldest = memories.stream()
                .map(Memory::getTimestamp)
                .min(Comparator.naturalOrder())
                .orElse(null);

        Instant newest = memories.stream()
                .map(Memory::getTimestamp)
                .max(Comparator.naturalOrder())
                .orElse(null);

        return Uni.createFrom().item(new MemoryStatistics(
                namespace, total, episodic, semantic, procedural, working,
                avgImportance, oldest, newest));
    }

    /**
     * Check if memory matches the given filters
     */
    private boolean matchesFilters(Memory memory, Map<String, Object> filters) {
        if (filters == null || filters.isEmpty()) {
            return true;
        }

        // Check namespace filter
        if (filters.containsKey("namespace")) {
            if (!memory.getNamespace().equals(filters.get("namespace"))) {
                return false;
            }
        }

        // Check type filter
        if (filters.containsKey("types")) {
            @SuppressWarnings("unchecked")
            List<String> allowedTypes = (List<String>) filters.get("types");
            if (!allowedTypes.contains(memory.getType().name())) {
                return false;
            }
        }

        // Check minimum importance
        if (filters.containsKey("minImportance")) {
            if (memory.getImportance() < (double) filters.get("minImportance")) {
                return false;
            }
        }

        // Check expiration
        if (memory.getExpiresAt() != null && memory.getExpiresAt().isBefore(Instant.now())) {
            return false; // Expired memory
        }

        return true;
    }

    /**
     * Calculate cosine similarity between two vectors
     */
    private double cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length) {
            return 0.0;
        }

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA == 0 || normB == 0) {
            return 0.0;
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Merge existing metadata with new metadata
     */
    private Map<String, Object> mergeMetadata(Map<String, Object> existing, Map<String, Object> updates) {
        Map<String, Object> merged = new HashMap<>(existing);
        merged.putAll(updates);
        return merged;
    }
}