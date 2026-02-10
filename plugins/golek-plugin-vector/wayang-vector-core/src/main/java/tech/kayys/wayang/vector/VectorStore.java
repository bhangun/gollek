package tech.kayys.wayang.vector;

import io.smallrye.mutiny.Uni;
import java.util.List;
import java.util.Map;

/**
 * Interface for Vector Store operations.
 */
public interface VectorStore {

    /**
     * Store entries in the vector store.
     *
     * @param entries list of entries to store
     * @return Uni<Void>
     */
    Uni<Void> store(List<VectorEntry> entries);

    /**
     * Search for similar entries.
     *
     * @param query the query parameters
     * @return Uni list of matching entries
     */
    Uni<List<VectorEntry>> search(VectorQuery query);

    /**
     * Search for similar entries with metadata filters.
     *
     * @param query the query parameters
     * @param filters metadata filters to apply
     * @return Uni list of matching entries
     */
    Uni<List<VectorEntry>> search(VectorQuery query, Map<String, Object> filters);

    /**
     * Delete entries by ID.
     *
     * @param ids list of IDs to delete
     * @return Uni<Void>
     */
    Uni<Void> delete(List<String> ids);

    /**
     * Delete entries by metadata filters.
     *
     * @param filters metadata filters to identify entries to delete
     * @return Uni<Void>
     */
    Uni<Void> deleteByFilters(Map<String, Object> filters);
}
