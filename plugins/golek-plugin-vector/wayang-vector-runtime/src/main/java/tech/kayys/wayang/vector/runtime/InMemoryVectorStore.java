package tech.kayys.wayang.vector.runtime;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.vector.AbstractVectorStore;
import tech.kayys.wayang.vector.VectorEntry;
import tech.kayys.wayang.vector.VectorQuery;
import tech.kayys.wayang.vector.VectorStore;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * In-memory implementation of VectorStore for development and testing purposes.
 */
public class InMemoryVectorStore extends AbstractVectorStore {

    private final Map<String, VectorEntry> store = new ConcurrentHashMap<>();

    @Override
    public Uni<Void> store(List<VectorEntry> entries) {
        for (VectorEntry entry : entries) {
            store.put(entry.id(), entry);
        }
        return Uni.createFrom().voidItem();
    }

    @Override
    public Uni<List<VectorEntry>> search(VectorQuery query) {
        List<VectorEntry> results = new ArrayList<>();

        for (VectorEntry entry : store.values()) {
            // Calculate cosine similarity between query vector and entry vector
            float similarity = cosineSimilarity(query.vector().toArray(new Float[0]),
                    entry.vector().toArray(new Float[0]));

            // Only include entries that meet the minimum score threshold
            if (similarity >= query.minScore()) {
                results.add(entry);
            }
        }

        // Sort by similarity (descending) and take top-k
        List<VectorEntry> sortedResults = results.stream()
                .sorted((a, b) -> {
                    float simA = cosineSimilarity(query.vector().toArray(new Float[0]),
                            a.vector().toArray(new Float[0]));
                    float simB = cosineSimilarity(query.vector().toArray(new Float[0]),
                            b.vector().toArray(new Float[0]));
                    return Float.compare(simB, simA); // Descending order
                })
                .limit(query.topK())
                .collect(Collectors.toList());

        return Uni.createFrom().item(sortedResults);
    }

    @Override
    public Uni<Void> delete(List<String> ids) {
        for (String id : ids) {
            store.remove(id);
        }
        return Uni.createFrom().voidItem();
    }

    /**
     * Calculate cosine similarity between two vectors.
     */
    private float cosineSimilarity(Float[] vectorA, Float[] vectorB) {
        if (vectorA.length != vectorB.length) {
            return 0.0f;
        }

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < vectorA.length; i++) {
            float a = vectorA[i];
            float b = vectorB[i];
            dotProduct += a * b;
            normA += a * a;
            normB += b * b;
        }

        if (normA == 0.0 || normB == 0.0) {
            return 0.0f;
        }

        return (float) (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
    }
}