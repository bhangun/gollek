package tech.kayys.wayang.vector.pgvector;

import io.quarkus.test.junit.QuarkusTest;
import io.smallrye.mutiny.Uni;
import io.vertx.mutiny.pgclient.PgPool;
import io.vertx.mutiny.sqlclient.SqlClient;
import io.vertx.mutiny.sqlclient.Tuple;
import jakarta.inject.Inject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.wayang.vector.VectorEntry;
import tech.kayys.wayang.vector.VectorQuery;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@QuarkusTest
class PgVectorStoreTest {

    @Inject
    PgVectorStore pgVectorStore;

    @Inject
    PgPool pgPool;

    @BeforeEach
    void setUp() {
        // Clean up any existing test data
        pgPool.query("TRUNCATE TABLE wayang_vector_entries").executeAndForget();
    }

    @Test
    void testInitialize() {
        Uni<Void> result = pgVectorStore.initialize();
        
        // Wait for initialization to complete
        result.await().indefinitely();
        
        // Verify the table was created by checking if it exists
        Boolean tableExists = pgPool.preparedQuery(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'wayang_vector_entries')")
                .execute()
                .map(rowSet -> rowSet.iterator().next().getBoolean(0))
                .await().indefinitely();
        
        assertTrue(tableExists, "Vector entries table should exist after initialization");
    }

    @Test
    void testStoreAndRetrieve() {
        // Initialize the store
        pgVectorStore.initialize().await().indefinitely();
        
        // Create test vector entries
        VectorEntry entry1 = new VectorEntry(
                "test-id-1",
                List.of(0.1f, 0.2f, 0.3f),
                "Test content 1",
                Map.of("category", "test", "source", "unittest")
        );
        
        VectorEntry entry2 = new VectorEntry(
                "test-id-2", 
                List.of(0.4f, 0.5f, 0.6f),
                "Test content 2",
                Map.of("category", "test", "source", "unittest")
        );

        // Store the entries
        pgVectorStore.store(List.of(entry1, entry2)).await().indefinitely();

        // Verify entries were stored by querying directly
        Long count = pgPool.preparedQuery("SELECT COUNT(*) FROM wayang_vector_entries WHERE id = $1")
                .execute(Tuple.of("test-id-1"))
                .map(rowSet -> rowSet.iterator().next().getLong(0))
                .await().indefinitely();

        assertEquals(1L, count, "Entry should be stored in the database");
    }

    @Test
    void testSearch() {
        // Initialize the store
        pgVectorStore.initialize().await().indefinitely();
        
        // Create and store test entries
        VectorEntry entry1 = new VectorEntry(
                "search-test-1",
                List.of(0.1f, 0.2f, 0.3f),
                "Apple pie recipe",
                Map.of("category", "food", "type", "recipe")
        );
        
        VectorEntry entry2 = new VectorEntry(
                "search-test-2", 
                List.of(0.9f, 0.8f, 0.7f),
                "Machine learning algorithm",
                Map.of("category", "technology", "type", "algorithm")
        );

        pgVectorStore.store(List.of(entry1, entry2)).await().indefinitely();

        // Perform a search with a similar vector to entry1
        VectorQuery query = new VectorQuery(List.of(0.15f, 0.25f, 0.35f), 5, 0.1f);
        List<VectorEntry> results = pgVectorStore.search(query).await().indefinitely();

        assertFalse(results.isEmpty(), "Search should return results");
        assertEquals("search-test-1", results.get(0).id(), "Most similar entry should be returned first");
    }

    @Test
    void testSearchWithFilters() {
        // Initialize the store
        pgVectorStore.initialize().await().indefinitely();
        
        // Create and store test entries with different categories
        VectorEntry entry1 = new VectorEntry(
                "filter-test-1",
                List.of(0.1f, 0.2f, 0.3f),
                "Technology article",
                Map.of("category", "tech", "author", "john")
        );
        
        VectorEntry entry2 = new VectorEntry(
                "filter-test-2", 
                List.of(0.9f, 0.8f, 0.7f),
                "Food recipe",
                Map.of("category", "food", "author", "jane")
        );

        pgVectorStore.store(List.of(entry1, entry2)).await().indefinitely();

        // Search with category filter
        VectorQuery query = new VectorQuery(List.of(0.15f, 0.25f, 0.35f), 5, 0.1f);
        Map<String, Object> filters = Map.of("category", "tech");
        List<VectorEntry> results = pgVectorStore.search(query, filters).await().indefinitely();

        assertFalse(results.isEmpty(), "Search with filter should return results");
        assertEquals("filter-test-1", results.get(0).id(), "Filtered result should match the filter criteria");
        assertEquals("tech", results.get(0).metadata().get("category"), "Result should have correct category");
    }

    @Test
    void testDeleteById() {
        // Initialize the store
        pgVectorStore.initialize().await().indefinitely();
        
        // Create and store a test entry
        VectorEntry entry = new VectorEntry(
                "delete-test-1",
                List.of(0.5f, 0.6f, 0.7f),
                "To be deleted",
                Map.of("status", "active")
        );

        pgVectorStore.store(List.of(entry)).await().indefinitely();

        // Verify it exists
        Long countBefore = pgPool.preparedQuery("SELECT COUNT(*) FROM wayang_vector_entries WHERE id = $1")
                .execute(Tuple.of("delete-test-1"))
                .map(rowSet -> rowSet.iterator().next().getLong(0))
                .await().indefinitely();

        assertEquals(1L, countBefore, "Entry should exist before deletion");

        // Delete the entry
        pgVectorStore.delete(List.of("delete-test-1")).await().indefinitely();

        // Verify it's gone
        Long countAfter = pgPool.preparedQuery("SELECT COUNT(*) FROM wayang_vector_entries WHERE id = $1")
                .execute(Tuple.of("delete-test-1"))
                .map(rowSet -> rowSet.iterator().next().getLong(0))
                .await().indefinitely();

        assertEquals(0L, countAfter, "Entry should be deleted");
    }

    @Test
    void testDeleteByFilters() {
        // Initialize the store
        pgVectorStore.initialize().await().indefinitely();
        
        // Create and store test entries with same category
        VectorEntry entry1 = new VectorEntry(
                "filter-delete-1",
                List.of(0.1f, 0.2f, 0.3f),
                "Tech article 1",
                Map.of("category", "tech", "status", "published")
        );
        
        VectorEntry entry2 = new VectorEntry(
                "filter-delete-2", 
                List.of(0.4f, 0.5f, 0.6f),
                "Tech article 2",
                Map.of("category", "tech", "status", "draft")
        );

        pgVectorStore.store(List.of(entry1, entry2)).await().indefinitely();

        // Verify both entries exist
        Long countBefore = pgPool.preparedQuery("SELECT COUNT(*) FROM wayang_vector_entries WHERE metadata->>'category' = $1")
                .execute(Tuple.of("tech"))
                .map(rowSet -> rowSet.iterator().next().getLong(0))
                .await().indefinitely();

        assertEquals(2L, countBefore, "Both tech entries should exist before deletion");

        // Delete by category filter
        Map<String, Object> filters = Map.of("category", "tech");
        pgVectorStore.deleteByFilters(filters).await().indefinitely();

        // Verify entries are gone
        Long countAfter = pgPool.preparedQuery("SELECT COUNT(*) FROM wayang_vector_entries WHERE metadata->>'category' = $1")
                .execute(Tuple.of("tech"))
                .map(rowSet -> rowSet.iterator().next().getLong(0))
                .await().indefinitely();

        assertEquals(0L, countAfter, "Tech entries should be deleted by filter");
    }

    @Test
    void testEmptyOperations() {
        // Initialize the store
        pgVectorStore.initialize().await().indefinitely();
        
        // Test storing empty list
        pgVectorStore.store(List.of()).await().indefinitely();
        
        // Test deleting empty list
        pgVectorStore.delete(List.of()).await().indefinitely();
        
        // Test search with no results
        VectorQuery query = new VectorQuery(List.of(0.0f, 0.0f, 0.0f), 5, 0.99f); // Very high threshold
        List<VectorEntry> results = pgVectorStore.search(query).await().indefinitely();
        
        assertTrue(results.isEmpty(), "Search with high threshold should return empty results");
    }
}