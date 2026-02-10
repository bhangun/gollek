package tech.kayys.wayang.vector.runtime;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.wayang.vector.VectorEntry;
import tech.kayys.wayang.vector.VectorQuery;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class InMemoryVectorStoreTest {

    private InMemoryVectorStore vectorStore;

    @BeforeEach
    void setUp() {
        vectorStore = new InMemoryVectorStore();
    }

    @Test
    void testStoreAndSearch() {
        // Create test entries
        VectorEntry entry1 = new VectorEntry(
            "1",
            List.of(0.1f, 0.2f, 0.3f),
            "Test content 1",
            Map.of("category", "test", "agentId", "agent1")
        );
        
        VectorEntry entry2 = new VectorEntry(
            "2", 
            List.of(0.4f, 0.5f, 0.6f),
            "Test content 2",
            Map.of("category", "test", "agentId", "agent2")
        );

        // Store entries
        vectorStore.store(List.of(entry1, entry2)).await().indefinitely();

        // Search with a similar vector
        VectorQuery query = new VectorQuery(List.of(0.1f, 0.2f, 0.3f), 5, 0.0f);
        List<VectorEntry> results = vectorStore.search(query).await().indefinitely();

        assertFalse(results.isEmpty());
        assertEquals("1", results.get(0).id()); // Should find the most similar entry
    }

    @Test
    void testSearchWithFilters() {
        // Create test entries
        VectorEntry entry1 = new VectorEntry(
            "1",
            List.of(0.1f, 0.2f, 0.3f),
            "Test content 1",
            Map.of("category", "test", "agentId", "agent1")
        );
        
        VectorEntry entry2 = new VectorEntry(
            "2", 
            List.of(0.4f, 0.5f, 0.6f),
            "Test content 2",
            Map.of("category", "test", "agentId", "agent2")
        );

        // Store entries
        vectorStore.store(List.of(entry1, entry2)).await().indefinitely();

        // Search with filters
        VectorQuery query = new VectorQuery(List.of(0.1f, 0.2f, 0.3f), 5, 0.0f);
        Map<String, Object> filters = Map.of("agentId", "agent1");
        List<VectorEntry> results = vectorStore.search(query, filters).await().indefinitely();

        assertEquals(1, results.size());
        assertEquals("1", results.get(0).id());
        assertEquals("agent1", results.get(0).metadata().get("agentId"));
    }

    @Test
    void testDeleteById() {
        // Create test entries
        VectorEntry entry1 = new VectorEntry(
            "1",
            List.of(0.1f, 0.2f, 0.3f),
            "Test content 1",
            Map.of("category", "test", "agentId", "agent1")
        );
        
        VectorEntry entry2 = new VectorEntry(
            "2", 
            List.of(0.4f, 0.5f, 0.6f),
            "Test content 2",
            Map.of("category", "test", "agentId", "agent2")
        );

        // Store entries
        vectorStore.store(List.of(entry1, entry2)).await().indefinitely();

        // Verify entries exist
        VectorQuery query = new VectorQuery(List.of(0.1f, 0.2f, 0.3f), 5, 0.0f);
        List<VectorEntry> results = vectorStore.search(query).await().indefinitely();
        assertEquals(2, results.size());

        // Delete one entry
        vectorStore.delete(List.of("1")).await().indefinitely();

        // Verify only one entry remains
        results = vectorStore.search(query).await().indefinitely();
        assertEquals(1, results.size());
        assertEquals("2", results.get(0).id());
    }

    @Test
    void testDeleteByFilters() {
        // Create test entries
        VectorEntry entry1 = new VectorEntry(
            "1",
            List.of(0.1f, 0.2f, 0.3f),
            "Test content 1",
            Map.of("category", "test", "agentId", "agent1")
        );
        
        VectorEntry entry2 = new VectorEntry(
            "2", 
            List.of(0.4f, 0.5f, 0.6f),
            "Test content 2",
            Map.of("category", "test", "agentId", "agent2")
        );

        // Store entries
        vectorStore.store(List.of(entry1, entry2)).await().indefinitely();

        // Verify entries exist
        VectorQuery query = new VectorQuery(List.of(0.1f, 0.2f, 0.3f), 5, 0.0f);
        List<VectorEntry> results = vectorStore.search(query).await().indefinitely();
        assertEquals(2, results.size());

        // Delete by filter
        Map<String, Object> filters = Map.of("agentId", "agent1");
        vectorStore.deleteByFilters(filters).await().indefinitely();

        // Verify only one entry remains
        results = vectorStore.search(query).await().indefinitely();
        assertEquals(1, results.size());
        assertEquals("2", results.get(0).id());
    }
}