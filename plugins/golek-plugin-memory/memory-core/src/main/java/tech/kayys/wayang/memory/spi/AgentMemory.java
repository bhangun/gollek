package tech.kayys.wayang.memory.spi;

import io.smallrye.mutiny.Uni;
import java.util.List;

/**
 * Interface for Agent Memory management.
 * Combines Short Term (Context) and Long Term (Vector/Semantic) memory.
 */
public interface AgentMemory {

    /**
     * Store an entry in the agent's memory.
     * Use metadata to distinguish between short-term and long-term if needed.
     *
     * @param agentId the agent identifier
     * @param entry   the memory entry
     * @return Uni<Void>
     */
    Uni<Void> store(String agentId, MemoryEntry entry);

    /**
     * Retrieve relevant memory entries based on a query.
     * Utilizes vector search for semantic relevance.
     *
     * @param agentId the agent identifier
     * @param query   the search query text
     * @param limit   maximum number of results
     * @return Uni list of relevant entries
     */
    Uni<List<MemoryEntry>> retrieve(String agentId, String query, int limit);

    /**
     * Get the current short-term context/history for the agent.
     *
     * @param agentId the agent identifier
     * @return Uni list of recent entries
     */
    Uni<List<MemoryEntry>> getContext(String agentId);

    /**
     * Clear memory for an agent.
     *
     * @param agentId the agent identifier
     * @return Uni<Void>
     */
    Uni<Void> clear(String agentId);
}
