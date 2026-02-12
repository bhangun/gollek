package tech.kayys.wayang.vector.runtime;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.vector.AbstractVectorStore;
import tech.kayys.wayang.vector.VectorEntry;
import tech.kayys.wayang.vector.VectorQuery;
import java.util.List;

/**
 * Placeholder implementation for PgVectorStore.
 * This would be replaced with actual PostgreSQL/pgvector implementation.
 */
public class PgVectorStore extends AbstractVectorStore {

    @Override
    public Uni<Void> store(List<VectorEntry> entries) {
        // Actual implementation would connect to PostgreSQL with pgvector extension
        throw new UnsupportedOperationException("PgVectorStore not yet implemented");
    }

    @Override
    public Uni<List<VectorEntry>> search(VectorQuery query) {
        // Actual implementation would perform vector similarity search in PostgreSQL
        throw new UnsupportedOperationException("PgVectorStore not yet implemented");
    }

    @Override
    public Uni<Void> delete(List<String> ids) {
        // Actual implementation would delete from PostgreSQL
        throw new UnsupportedOperationException("PgVectorStore not yet implemented");
    }
}