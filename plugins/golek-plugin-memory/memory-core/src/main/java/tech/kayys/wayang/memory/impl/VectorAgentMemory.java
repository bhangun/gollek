package tech.kayys.wayang.memory.impl;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.wayang.memory.model.Memory;
import tech.kayys.wayang.memory.service.VectorMemoryStore;
import tech.kayys.wayang.memory.spi.AgentMemory;
import tech.kayys.wayang.memory.spi.EmbeddingService;
import tech.kayys.wayang.memory.spi.MemoryEntry;
import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.memory.model.ScoredMemory;

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
    VectorMemoryStore vectorMemoryStore;

    @Inject
    EmbeddingService embeddingService;

    @Override
    public Uni<Void> store(String agentId, MemoryEntry entry) {
        if (entry.content() == null || entry.content().isBlank()) {
            return Uni.createFrom().voidItem();
        }

        return embeddingService.embed(entry.content())
                .flatMap(vector -> {
                    Memory memory = Memory.builder()
                            .id(entry.id() != null ? entry.id() : UUID.randomUUID().toString())
                            .content(entry.content())
                            .embedding(toFloatArray(vector))
                            .addMetadata("agentId", agentId)
                            .addMetadata("timestamp", entry.timestamp().toString())
                            .build();

                    return vectorMemoryStore.store(memory)
                            .map(id -> null); // Convert to Void
                });
    }

    @Override
    public Uni<List<MemoryEntry>> retrieve(String agentId, String query, int limit) {
        return embeddingService.embed(query)
                .flatMap(vector -> {
                    // Apply agentId filter to search only memories for this agent
                    Map<String, Object> filters = Map.of("agentId", agentId);
                    return vectorMemoryStore.search(toFloatArray(vector), limit, 0.0, filters);
                })
                .map(scoredMemories -> scoredMemories.stream()
                        .map(scoredMemory -> toMemoryEntry(scoredMemory.getMemory()))
                        .collect(Collectors.toList()));
    }

    @Override
    public Uni<List<MemoryEntry>> getContext(String agentId) {
        // This is a naive implementation of context - just retrieving recent items by
        // time?
        // Or maybe ShortTerm memory is stored differently.
        // For now, returning empty list as placeholders or we could implement a
        // separate store.
        return Uni.createFrom().item(Collections.emptyList());
    }

    @Override
    public Uni<Void> clear(String agentId) {
        // Delete all memories for this agent using namespace (which contains agentId)
        return vectorMemoryStore.deleteNamespace(agentId)
                .replaceWithVoid();
    }

    private MemoryEntry toMemoryEntry(Memory memory) {
        Instant timestamp = Instant.now();
        if (memory.getMetadata().containsKey("timestamp")) {
            try {
                timestamp = Instant.parse((String) memory.getMetadata().get("timestamp"));
            } catch (Exception e) {
                // ignore
            }
        }
        return new MemoryEntry(memory.getId(), memory.getContent(), timestamp, memory.getMetadata());
    }

    /**
     * Convert List<Float> to float[]
     */
    private float[] toFloatArray(List<Float> list) {
        if (list == null) return new float[0];
        float[] result = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            result[i] = list.get(i);
        }
        return result;
    }
}
