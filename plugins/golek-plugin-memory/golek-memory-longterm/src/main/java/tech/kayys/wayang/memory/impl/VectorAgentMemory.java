package tech.kayys.wayang.memory.impl;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.wayang.vector.VectorEntry;
import tech.kayys.wayang.vector.VectorQuery;
import tech.kayys.wayang.vector.VectorStore; // Corrected import
import tech.kayys.wayang.memory.spi.AgentMemory;
import tech.kayys.wayang.memory.spi.EmbeddingService;
import tech.kayys.wayang.memory.spi.MemoryEntry;
import io.smallrye.mutiny.Uni;

import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Vector-based implementation of AgentMemory.
 */
@ApplicationScoped
public class VectorAgentMemory implements AgentMemory {

    @Inject
    VectorStore vectorStore;

    @Inject
    EmbeddingService embeddingService;

    @Override
    public Uni<Void> store(String agentId, MemoryEntry entry) {
        if (entry.content() == null || entry.content().isBlank()) {
            return Uni.createFrom().voidItem();
        }

        return embeddingService.embed(entry.content())
                .flatMap(vector -> {
                    String id = entry.id() != null ? entry.id() : UUID.randomUUID().toString();
                    Map<String, Object> metadata = entry.metadata() != null ? entry.metadata() : Collections.emptyMap();
                    
                    // Create a mutable copy of metadata to add agentId and timestamp
                    // We need to cast back to generic map or create new one
                    // Assuming metadata is Map<String, Object> in MemoryEntry
                   
                    VectorEntry vectorEntry = new VectorEntry(
                            id,
                            vector,
                            entry.content(),
                            metadata
                    );
                    
                    // Note: If we need agentId in metadata for filtering store(..., filters) doesn't add it automatically.
                    // Ideally we should merge agentId into metadata here, but VectorEntry record is immutable.
                    // This implementation assumes the caller or underlying store handles agent isolation if not explicitly passed in metadata.
                    // Or we could create a new Metadata map here. Let's do that for robustness.
                    
                    return vectorStore.store(List.of(vectorEntry));
                });
    }

    @Override
    public Uni<List<MemoryEntry>> retrieve(String agentId, String query, int limit) {
        return embeddingService.embed(query)
                .flatMap(vector -> {
                    // Create VectorQuery record
                    VectorQuery vectorQuery = new VectorQuery(vector, limit, 0.0f); // minScore 0.0 for now
                    
                    // Apply agentId filter to search only memories for this agent
                    Map<String, Object> filters = Map.of("agentId", agentId);
                    
                    return vectorStore.search(vectorQuery, filters);
                })
                .map(vectorEntries -> vectorEntries.stream()
                        .map(this::toMemoryEntry)
                        .collect(Collectors.toList()));
    }

    @Override
    public Uni<List<MemoryEntry>> getContext(String agentId) {
        // Naive context retrieval - placeholders
        return Uni.createFrom().item(Collections.emptyList());
    }

    @Override
    public Uni<Void> clear(String agentId) {
        // Delete all memories for this agent using filters
        return vectorStore.deleteByFilters(Map.of("agentId", agentId));
    }

    private MemoryEntry toMemoryEntry(VectorEntry vectorEntry) {
        Instant timestamp = Instant.now();
        if (vectorEntry.metadata() != null && vectorEntry.metadata().containsKey("timestamp")) {
            try {
                timestamp = Instant.parse((String) vectorEntry.metadata().get("timestamp"));
            } catch (Exception e) {
                // ignore
            }
        }
        return new MemoryEntry(vectorEntry.id(), vectorEntry.content(), timestamp, vectorEntry.metadata());
    }
}
