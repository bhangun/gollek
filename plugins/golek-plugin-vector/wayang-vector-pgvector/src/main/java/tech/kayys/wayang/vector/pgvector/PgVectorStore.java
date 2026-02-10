package tech.kayys.wayang.vector.pgvector;

import io.quarkus.hibernate.reactive.panache.Panache;
import io.smallrye.mutiny.Uni;
import io.vertx.mutiny.pgclient.PgPool;
import io.vertx.mutiny.sqlclient.Row;
import io.vertx.mutiny.sqlclient.Tuple;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.vector.AbstractVectorStore;
import tech.kayys.wayang.vector.VectorEntry;
import tech.kayys.wayang.vector.VectorQuery;

import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Production-grade vector store implementation using PostgreSQL with pgvector extension.
 */
@ApplicationScoped
public class PgVectorStore extends AbstractVectorStore {

    private static final Logger LOG = LoggerFactory.getLogger(PgVectorStore.class);

    @Inject
    PgPool pgPool;

    @ConfigProperty(name = "wayang.vector.pgvector.dimension", defaultValue = "1536")
    int vectorDimension;

    @ConfigProperty(name = "wayang.vector.pgvector.index.type", defaultValue = "hnsw")
    String indexType; // hnsw or ivfflat

    /**
     * Initialize database schema and pgvector extension
     */
    public Uni<Void> initialize() {
        LOG.info("Initializing PostgreSQL vector store with dimension: {} and index type: {}", vectorDimension, indexType);

        String createExtensionSql = "CREATE EXTENSION IF NOT EXISTS vector;";
        String createTableSql = """
                CREATE TABLE IF NOT EXISTS wayang_vector_entries (
                    id VARCHAR(255) PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(%d),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                -- Vector similarity index (HNSW for better performance)
                DROP INDEX IF EXISTS idx_vector_entries_embedding_%s;
                CREATE INDEX idx_vector_entries_embedding_%s
                    ON wayang_vector_entries USING %s (embedding vector_cosine_ops);

                -- Index for faster metadata lookups
                CREATE INDEX IF NOT EXISTS idx_vector_entries_metadata_gin
                    ON wayang_vector_entries USING GIN (metadata);
                """.formatted(vectorDimension, indexType, indexType, indexType);

        return pgPool.query(createExtensionSql)
                .execute()
                .onItem().transformToUni(result -> pgPool.query(createTableSql).execute())
                .replaceWithVoid()
                .invoke(() -> LOG.info("Vector store initialized"));
    }

    @Override
    public Uni<Void> store(List<VectorEntry> entries) {
        LOG.debug("Storing {} vector entries", entries.size());

        if (entries.isEmpty()) {
            return Uni.createFrom().voidItem();
        }

        return Panache.withTransaction(() -> {
            List<Uni<Void>> stores = entries.stream()
                    .map(this::storeSingle)
                    .collect(Collectors.toList());

            return Uni.join().all(stores).andFailFast().replaceWithVoid();
        });
    }

    /**
     * Store a single vector entry
     */
    private Uni<Void> storeSingle(VectorEntry entry) {
        String sql = """
                INSERT INTO wayang_vector_entries (id, content, embedding, metadata)
                VALUES ($1, $2, $3::vector, $4::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """;

        Tuple params = Tuple.of(
                entry.id(),
                entry.content(),
                vectorToString(entry.vector()),
                toJsonb(entry.metadata())
        );

        return pgPool.preparedQuery(sql)
                .execute(params)
                .replaceWithVoid();
    }

    @Override
    public Uni<List<VectorEntry>> search(VectorQuery query) {
        LOG.debug("Vector search with topK: {}, minScore: {}", query.topK(), query.minScore());

        String sql = """
                SELECT id, content, embedding::text, metadata
                FROM wayang_vector_entries
                WHERE (embedding <=> $1::vector) <= (1 - $2::float)
                ORDER BY embedding <=> $1::vector ASC
                LIMIT $3
                """;

        Tuple params = Tuple.of(
                vectorToString(query.vector()),
                query.minScore(),
                query.topK()
        );

        return pgPool.preparedQuery(sql)
                .execute(params)
                .map(rowSet -> {
                    List<VectorEntry> results = new ArrayList<>();
                    for (Row row : rowSet) {
                        VectorEntry entry = rowToVectorEntry(row);
                        results.add(entry);
                    }
                    LOG.debug("Found {} results for vector search", results.size());
                    return results;
                });
    }

    @Override
    public Uni<List<VectorEntry>> search(VectorQuery query, Map<String, Object> filters) {
        LOG.debug("Vector search with filters: {} and topK: {}, minScore: {}", filters, query.topK(), query.minScore());

        // Build dynamic query based on filters
        StringBuilder sql = new StringBuilder("""
                SELECT id, content, embedding::text, metadata
                FROM wayang_vector_entries
                WHERE (embedding <=> $1::vector) <= (1 - $2::float)
                """);

        List<Object> params = new ArrayList<>();
        params.add(vectorToString(query.vector()));
        params.add(query.minScore());
        int paramIndex = 3;

        // Add metadata filters
        if (filters != null && !filters.isEmpty()) {
            for (Map.Entry<String, Object> filter : filters.entrySet()) {
                sql.append(" AND metadata->>$").append(paramIndex++).append(" = $").append(paramIndex);
                params.add(filter.getKey());
                params.add(String.valueOf(filter.getValue()));
            }
        }

        // Order and limit
        sql.append(" ORDER BY embedding <=> $1::vector ASC LIMIT $").append(paramIndex);

        params.add(query.topK());

        return pgPool.preparedQuery(sql.toString())
                .execute(Tuple.wrap(params))
                .map(rowSet -> {
                    List<VectorEntry> results = new ArrayList<>();
                    for (Row row : rowSet) {
                        VectorEntry entry = rowToVectorEntry(row);
                        results.add(entry);
                    }
                    LOG.debug("Found {} results for filtered vector search", results.size());
                    return results;
                });
    }

    @Override
    public Uni<Void> delete(List<String> ids) {
        LOG.debug("Deleting {} vector entries", ids.size());

        if (ids.isEmpty()) {
            return Uni.createFrom().voidItem();
        }

        String sql = "DELETE FROM wayang_vector_entries WHERE id = ANY($1)";
        Tuple params = Tuple.of(ids.toArray(new String[0]));

        return pgPool.preparedQuery(sql)
                .execute(params)
                .replaceWithVoid();
    }

    @Override
    public Uni<Void> deleteByFilters(Map<String, Object> filters) {
        LOG.debug("Deleting vector entries by filters: {}", filters);

        if (filters == null || filters.isEmpty()) {
            return Uni.createFrom().voidItem();
        }

        StringBuilder sql = new StringBuilder("DELETE FROM wayang_vector_entries WHERE ");
        List<Object> params = new ArrayList<>();
        boolean first = true;

        for (Map.Entry<String, Object> filter : filters.entrySet()) {
            if (!first) {
                sql.append(" AND ");
            }
            sql.append("metadata->>$").append(first ? 1 : 3).append(" = $").append(first ? 2 : 4);
            params.add(filter.getKey());
            params.add(String.valueOf(filter.getValue()));
            first = false;
        }

        return pgPool.preparedQuery(sql.toString())
                .execute(Tuple.wrap(params))
                .replaceWithVoid();
    }

    /**
     * Convert row to VectorEntry
     */
    private VectorEntry rowToVectorEntry(Row row) {
        return new VectorEntry(
                row.getString("id"),
                stringToVector(row.getString("embedding")),
                row.getString("content"),
                fromJsonb(row.getString("metadata"))
        );
    }

    /**
     * Convert float list to PostgreSQL vector format
     */
    private String vectorToString(List<Float> vector) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < vector.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append(vector.get(i));
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Convert PostgreSQL vector string to float list
     */
    private List<Float> stringToVector(String vectorStr) {
        if (vectorStr == null || vectorStr.isEmpty()) {
            return new ArrayList<>();
        }

        // Remove brackets and split
        String cleaned = vectorStr.substring(1, vectorStr.length() - 1);
        String[] parts = cleaned.split(",");

        List<Float> vector = new ArrayList<>();
        for (String part : parts) {
            if (!part.trim().isEmpty()) {
                vector.add(Float.parseFloat(part.trim()));
            }
        }

        return vector;
    }

    /**
     * Convert map to JSONB string
     */
    private String toJsonb(Map<String, Object> map) {
        try {
            return new com.fasterxml.jackson.databind.ObjectMapper()
                    .writeValueAsString(map);
        } catch (Exception e) {
            LOG.error("Failed to convert map to JSONB", e);
            return "{}";
        }
    }

    /**
     * Convert JSONB string to map
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> fromJsonb(String jsonb) {
        try {
            if (jsonb == null || jsonb.trim().isEmpty() || jsonb.equals("{}")) {
                return new HashMap<>();
            }
            return new com.fasterxml.jackson.databind.ObjectMapper()
                    .readValue(jsonb, Map.class);
        } catch (Exception e) {
            LOG.error("Failed to parse JSONB", e);
            return new HashMap<>();
        }
    }
}