package tech.kayys.wayang.vector.runtime;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.vector.AbstractVectorStore;
import tech.kayys.wayang.vector.VectorEntry;
import tech.kayys.wayang.vector.VectorQuery;
import java.util.List;

/**
 * Placeholder implementation for QdrantVectorStore.
 * This would be replaced with actual Qdrant implementation.
 */
public class QdrantVectorStore extends AbstractVectorStore {

    @Override
    public Uni<Void> store(List<VectorEntry> entries) {
        // Actual implementation would connect to Qdrant
        throw new UnsupportedOperationException("QdrantVectorStore not yet implemented");
    }

    @Override
    public Uni<List<VectorEntry>> search(VectorQuery query) {
        // Actual implementation would perform vector similarity search in Qdrant
        throw new UnsupportedOperationException("QdrantVectorStore not yet implemented");
    }

    @Override
    public Uni<Void> delete(List<String> ids) {
        // Actual implementation would delete from Qdrant
        throw new UnsupportedOperationException("QdrantVectorStore not yet implemented");
    }
}