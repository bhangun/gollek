package tech.kayys.golek.provider.core.memory;

import io.smallrye.mutiny.Uni;

import java.util.List;
import java.util.Optional;

/**
 * Core memory service interface.
 * Provides CRUD operations and advanced querying for all memory types.
 */
public interface MemoryService {

    /**
     * Store a memory entry
     */
    Uni<MemoryEntry> store(MemoryEntry entry);

    /**
     * Store multiple memory entries in batch
     */
    Uni<List<MemoryEntry>> storeBatch(List<MemoryEntry> entries);

    /**
     * Retrieve memory by ID
     */
    Uni<Optional<MemoryEntry>> retrieve(String id);

    /**
     * Query memories
     */
    Uni<List<MemoryEntry>> query(MemoryQuery query);

    /**
     * Update memory entry
     */
    Uni<MemoryEntry> update(MemoryEntry entry);

    /**
     * Delete memory by ID
     */
    Uni<Boolean> delete(String id);

    /**
     * Delete memories matching query
     */
    Uni<Integer> deleteMatching(MemoryQuery query);

    /**
     * Get total memory count
     */
    Uni<Long> count(MemoryQuery query);

    /**
     * Consolidate memories (merge similar entries)
     */
    Uni<List<MemoryEntry>> consolidate(MemoryQuery query);

    /**
     * Prune expired and low-importance memories
     */
    Uni<Integer> prune(MemoryQuery query);

    /**
     * Clear all memories for tenant
     */
    Uni<Void> clearAll(String tenantId);
}