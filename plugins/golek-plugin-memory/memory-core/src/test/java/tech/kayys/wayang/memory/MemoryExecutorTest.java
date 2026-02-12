package tech.kayys.wayang.memory;

import io.quarkus.test.junit.QuarkusTest;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import tech.kayys.wayang.memory.model.*;
import tech.kayys.wayang.memory.service.*;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Memory executor tests
 */
@QuarkusTest
public class MemoryExecutorTest {

    @Inject
    VectorMemoryStore memoryStore;

    @Inject
    EmbeddingServiceFactory embeddingFactory;

    @Inject
    ContextEngineeringService contextService;

    @Test
    public void testStoreAndRetrieveMemory() {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        String content = "Test memory content";

        String memoryId = embeddingService.embed(content)
                .flatMap(embedding -> {
                    Memory memory = Memory.builder()
                            .namespace("test")
                            .content(content)
                            .embedding(embedding)
                            .type(MemoryType.EPISODIC)
                            .importance(0.8)
                            .build();

                    return memoryStore.store(memory);
                })
                .await().indefinitely();

        assertNotNull(memoryId);

        Memory retrieved = memoryStore.retrieve(memoryId)
                .await().indefinitely();

        assertNotNull(retrieved);
        assertEquals(content, retrieved.getContent());
        assertEquals(MemoryType.EPISODIC, retrieved.getType());
        assertEquals(0.8, retrieved.getImportance(), 0.01);
    }

    @Test
    public void testSemanticSearch() {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        // Store some test memories
        List<String> contents = List.of(
                "Customer had issue with product quality",
                "Payment processing failed due to network error",
                "User requested refund for damaged item");

        Uni.join().all(
                contents.stream()
                        .map(content -> storeTestMemory(content, embeddingService))
                        .toList())
                .andFailFast()
                .await().indefinitely();

        // Search for similar
        String query = "Product quality complaint";

        List<ScoredMemory> results = embeddingService.embed(query)
                .flatMap(queryEmbedding -> memoryStore.search(
                        queryEmbedding,
                        3,
                        0.0,
                        Map.of("namespace", "test")))
                .await().indefinitely();

        assertNotNull(results);
        assertFalse(results.isEmpty());

        // First result should be most similar
        assertTrue(results.get(0).getScore() > 0);
    }

    @Test
    public void testContextEngineering() {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        // Store knowledge base
        List<String> knowledge = List.of(
                "To process refund: verify order, check policy, execute payment reversal",
                "Standard refund policy allows returns within 30 days",
                "Premium customers get expedited refund processing");

        Uni.join().all(
                knowledge.stream()
                        .map(k -> storeTestMemory(k, embeddingService))
                        .toList())
                .andFailFast()
                .await().indefinitely();

        // Build context
        String query = "How to handle refund request?";

        ContextConfig config = ContextConfig.builder()
                .maxMemories(3)
                .systemPrompt("You are a support assistant")
                .includeMetadata(false)
                .build();

        EngineerContext context = contextService
                .buildContext(query, "test", config)
                .await().indefinitely();

        assertNotNull(context);
        assertTrue(context.getTotalTokens() > 0);
        assertFalse(context.getSections().isEmpty());

        String prompt = context.toPrompt();
        assertNotNull(prompt);
        assertTrue(prompt.contains(query));
    }

    @Test
    public void testMemoryMetadata() {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        Memory memory = embeddingService.embed("Test with metadata")
                .map(embedding -> Memory.builder()
                        .namespace("test")
                        .content("Test with metadata")
                        .embedding(embedding)
                        .type(MemoryType.SEMANTIC)
                        .importance(0.7)
                        .addMetadata("category", "testing")
                        .addMetadata("priority", "high")
                        .build())
                .await().indefinitely();

        String memoryId = memoryStore.store(memory)
                .await().indefinitely();

        // Update metadata
        Map<String, Object> newMetadata = Map.of("status", "processed");

        Memory updated = memoryStore.updateMetadata(memoryId, newMetadata)
                .await().indefinitely();

        assertNotNull(updated);
        assertEquals("processed", updated.getMetadata().get("status"));
        assertEquals("testing", updated.getMetadata().get("category"));
    }

    @Test
    public void testMemoryStatistics() {
        // Store different types of memories
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        Uni.join().all(
                storeTestMemory("Episodic 1", MemoryType.EPISODIC, embeddingService),
                storeTestMemory("Episodic 2", MemoryType.EPISODIC, embeddingService),
                storeTestMemory("Semantic 1", MemoryType.SEMANTIC, embeddingService),
                storeTestMemory("Working 1", MemoryType.WORKING, embeddingService)).andFailFast()
                .await().indefinitely();

        MemoryStatistics stats = memoryStore.getStatistics("test")
                .await().indefinitely();

        assertNotNull(stats);
        assertTrue(stats.getTotalMemories() >= 4);
        assertTrue(stats.getEpisodicCount() >= 2);
        assertTrue(stats.getSemanticCount() >= 1);
        assertTrue(stats.getWorkingCount() >= 1);
    }

    // Helper methods

    private Uni<String> storeTestMemory(String content, EmbeddingService embeddingService) {
        return storeTestMemory(content, MemoryType.EPISODIC, embeddingService);
    }

    private Uni<String> storeTestMemory(
            String content,
            MemoryType type,
            EmbeddingService embeddingService) {

        return embeddingService.embed(content)
                .flatMap(embedding -> {
                    Memory memory = Memory.builder()
                            .namespace("test")
                            .content(content)
                            .embedding(embedding)
                            .type(type)
                            .importance(0.6)
                            .build();

                    return memoryStore.store(memory);
                });
    }
}
