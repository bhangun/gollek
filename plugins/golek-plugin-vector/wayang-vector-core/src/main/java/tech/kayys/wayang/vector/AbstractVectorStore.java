package tech.kayys.wayang.vector;

import io.smallrye.mutiny.Uni;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Abstract base implementation of VectorStore with common functionality.
 */
public abstract class AbstractVectorStore implements VectorStore {

    @Override
    public Uni<List<VectorEntry>> search(VectorQuery query, Map<String, Object> filters) {
        // Default implementation: perform search and then filter results in memory
        // Subclasses should override this with native filtering if possible for better performance
        return search(query)
                .map(entries -> entries.stream()
                        .filter(entry -> matchesFilters(entry, filters))
                        .collect(Collectors.toList()));
    }

    @Override
    public Uni<Void> deleteByFilters(Map<String, Object> filters) {
        // Default implementation: retrieve all entries matching filters and delete by ID
        // Subclasses should override this with native filtering if possible for better performance
        VectorQuery query = new VectorQuery(List.of(), Integer.MAX_VALUE, 0.0f); // Get all entries
        return search(query, filters)
                .flatMap(entries -> {
                    if (entries.isEmpty()) {
                        return Uni.createFrom().voidItem();
                    }
                    List<String> ids = entries.stream()
                            .map(VectorEntry::id)
                            .collect(Collectors.toList());
                    return delete(ids);
                });
    }

    /**
     * Helper method to check if a VectorEntry matches the given filters.
     */
    protected boolean matchesFilters(VectorEntry entry, Map<String, Object> filters) {
        if (filters == null || filters.isEmpty()) {
            return true;
        }

        Map<String, Object> metadata = entry.metadata();
        if (metadata == null) {
            return false;
        }

        for (Map.Entry<String, Object> filter : filters.entrySet()) {
            String key = filter.getKey();
            Object expectedValue = filter.getValue();
            Object actualValue = metadata.get(key);

            if (expectedValue == null) {
                if (actualValue != null) {
                    return false;
                }
            } else if (!expectedValue.equals(actualValue)) {
                return false;
            }
        }

        return true;
    }
}