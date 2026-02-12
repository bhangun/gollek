package tech.kayys.wayang.memory.examples;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.wayang.memory.model.*;
import tech.kayys.wayang.memory.service.*;

import java.time.Duration;
import java.time.Instant;
import java.util.*;

/**
 * Comprehensive examples demonstrating the memory executor capabilities.
 */
@ApplicationScoped
public class MemoryExecutorExamples {

    private static final Logger LOG = LoggerFactory.getLogger(MemoryExecutorExamples.class);

    @Inject
    VectorMemoryStore memoryStore;

    @Inject
    EmbeddingServiceFactory embeddingFactory;

    @Inject
    ContextEngineeringService contextService;

    /**
     * Example 1: Store and Retrieve Memories
     */
    public Uni<Void> example1_BasicMemoryOperations() {
        LOG.info("=== Example 1: Basic Memory Operations ===");

        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        // Create a memory about a customer interaction
        String content = """
                Customer John Doe contacted support regarding order #12345.
                Issue: Product arrived damaged.
                Resolution: Full refund processed, replacement shipped.
                Customer satisfaction: High
                """;

        return embeddingService.embed(content)
                .flatMap(embedding -> {
                    Memory memory = Memory.builder()
                            .namespace("customer-support")
                            .content(content)
                            .embedding(embedding)
                            .type(MemoryType.EPISODIC)
                            .importance(0.8)
                            .addMetadata("customerId", "CUST-001")
                            .addMetadata("orderId", "ORD-12345")
                            .addMetadata("category", "refund")
                            .addMetadata("sentiment", "positive")
                            .build();

                    return memoryStore.store(memory);
                })
                .flatMap(memoryId -> {
                    LOG.info("Stored memory: {}", memoryId);

                    // Retrieve the memory
                    return memoryStore.retrieve(memoryId);
                })
                .invoke(retrieved -> {
                    LOG.info("Retrieved memory: {}", retrieved.getContent());
                    LOG.info("Importance: {}", retrieved.getImportance());
                    LOG.info("Metadata: {}", retrieved.getMetadata());
                })
                .replaceWithVoid();
    }

    /**
     * Example 2: Semantic Search
     */
    public Uni<Void> example2_SemanticSearch() {
        LOG.info("=== Example 2: Semantic Search ===");

        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        // First, populate with some memories
        List<String> customerIssues = List.of(
                "Product arrived damaged, customer wants refund",
                "Delivery delayed by 2 days, customer upset",
                "Wrong item shipped, customer needs replacement",
                "Product quality excellent, customer very happy",
                "Customer cannot login to account, needs password reset");

        return Uni.join().all(
                customerIssues.stream()
                        .map(issue -> storeCustomerIssue(issue))
                        .toList())
                .andFailFast()
                .flatMap(stored -> {
                    LOG.info("Stored {} customer issues", stored.size());

                    // Now search for similar issues
                    String query = "Customer received broken product";

                    return embeddingService.embed(query)
                            .flatMap(queryEmbedding -> memoryStore.search(
                                    queryEmbedding,
                                    3, // Top 3 results
                                    0.5, // Min similarity
                                    Map.of("namespace", "customer-support")));
                })
                .invoke(results -> {
                    LOG.info("Found {} similar issues:", results.size());
                    results.forEach(scored -> {
                        LOG.info("  - Score: {:.3f} | {}",
                                scored.getScore(),
                                scored.getMemory().getContent().substring(0, 50) + "...");
                    });
                })
                .replaceWithVoid();
    }

    /**
     * Example 3: Context Engineering
     */
    public Uni<Void> example3_ContextEngineering() {
        LOG.info("=== Example 3: Context Engineering ===");

        // Populate knowledge base
        List<String> knowledgeArticles = List.of(
                "To process a refund: 1) Verify order number, 2) Check refund policy, 3) Process via payment system",
                "Damaged products qualify for full refund within 30 days of delivery",
                "Replacement shipping is free for damaged items",
                "Customer satisfaction is our top priority",
                "All refunds are processed within 3-5 business days");

        return Uni.join().all(
                knowledgeArticles.stream()
                        .map(article -> storeKnowledgeArticle(article))
                        .toList())
                .andFailFast()
                .flatMap(stored -> {
                    LOG.info("Stored {} knowledge articles", stored.size());

                    // Build context for a customer support query
                    String query = "How do I handle a damaged product complaint?";

                    ContextConfig config = ContextConfig.builder()
                            .maxMemories(3)
                            .systemPrompt("You are a helpful customer support assistant.")
                            .taskInstructions("Provide clear step-by-step guidance.")
                            .memoryTypes(List.of(MemoryType.SEMANTIC, MemoryType.PROCEDURAL))
                            .includeMetadata(false)
                            .build();

                    return contextService.buildContext(query, "customer-support", config);
                })
                .invoke(context -> {
                    LOG.info("Context built successfully:");
                    LOG.info("  Total tokens: {}", context.getTotalTokens());
                    LOG.info("  Utilization: {:.1f}%", context.getUtilization() * 100);
                    LOG.info("  Sections: {}", context.getSections().size());

                    LOG.info("\n=== Generated Prompt ===");
                    LOG.info(context.toPrompt());
                })
                .replaceWithVoid();
    }

    /**
     * Example 4: Hybrid Search (Semantic + Keyword)
     */
    public Uni<Void> example4_HybridSearch() {
        LOG.info("=== Example 4: Hybrid Search ===");

        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        String query = "refund policy for damaged items";
        List<String> keywords = List.of("refund", "damaged", "policy");

        return embeddingService.embed(query)
                .flatMap(queryEmbedding -> memoryStore.hybridSearch(
                        queryEmbedding,
                        keywords,
                        5,
                        0.7 // 70% semantic, 30% keyword
                ))
                .invoke(results -> {
                    LOG.info("Hybrid search results:");
                    results.forEach(scored -> {
                        LOG.info("  Combined Score: {:.3f}", scored.getScore());
                        LOG.info("    Breakdown: {}", scored.getScoreBreakdown());
                        LOG.info("    Content: {}...\n",
                                scored.getMemory().getContent().substring(0, 60));
                    });
                })
                .replaceWithVoid();
    }

    /**
     * Example 5: Memory with Temporal Decay
     */
    public Uni<Void> example5_TemporalDecay() {
        LOG.info("=== Example 5: Temporal Decay ===");

        // Create memories at different times
        Instant now = Instant.now();

        List<Uni<String>> memoryOps = new ArrayList<>();

        // Recent memory (1 hour ago)
        memoryOps.add(createTimedMemory(
                "Recent customer interaction - very helpful",
                now.minus(Duration.ofHours(1)),
                0.8));

        // Older memory (1 day ago)
        memoryOps.add(createTimedMemory(
                "Yesterday's interaction - also helpful",
                now.minus(Duration.ofDays(1)),
                0.8));

        // Very old memory (30 days ago)
        memoryOps.add(createTimedMemory(
                "Old interaction from last month",
                now.minus(Duration.ofDays(30)),
                0.8));

        return Uni.join().all(memoryOps).andFailFast()
                .flatMap(memoryIds -> {
                    // Retrieve all and show decayed importance
                    return memoryStore.retrieveBatch(memoryIds);
                })
                .invoke(memories -> {
                    LOG.info("Memory importance with temporal decay (rate=0.001):");

                    memories.forEach(memory -> {
                        Duration age = Duration.between(memory.getTimestamp(), Instant.now());
                        double decayed = memory.getDecayedImportance(0.001);

                        LOG.info("  Age: {} hours | Original: {:.3f} | Decayed: {:.3f}",
                                age.toHours(),
                                memory.getImportance(),
                                decayed);
                    });
                })
                .replaceWithVoid();
    }

    /**
     * Example 6: Batch Operations
     */
    public Uni<Void> example6_BatchOperations() {
        LOG.info("=== Example 6: Batch Operations ===");

        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        List<String> contents = List.of(
                "Customer inquiry about shipping times",
                "Product return request received",
                "Payment processing issue reported",
                "Account registration problem",
                "Newsletter subscription confirmed");

        // Batch embed
        return embeddingService.embedBatch(contents)
                .flatMap(embeddings -> {
                    LOG.info("Generated {} embeddings in batch", embeddings.size());

                    // Create memories
                    List<Memory> memories = new ArrayList<>();
                    for (int i = 0; i < contents.size(); i++) {
                        memories.add(Memory.builder()
                                .namespace("customer-support")
                                .content(contents.get(i))
                                .embedding(embeddings.get(i))
                                .type(MemoryType.EPISODIC)
                                .importance(0.5 + (i * 0.1))
                                .build());
                    }

                    // Batch store
                    return memoryStore.storeBatch(memories);
                })
                .invoke(memoryIds -> {
                    LOG.info("Stored {} memories in batch", memoryIds.size());
                    memoryIds.forEach(id -> LOG.info("  - {}", id));
                })
                .replaceWithVoid();
    }

    /**
     * Example 7: Memory Statistics
     */
    public Uni<Void> example7_MemoryStatistics() {
        LOG.info("=== Example 7: Memory Statistics ===");

        return memoryStore.getStatistics("customer-support")
                .invoke(stats -> {
                    LOG.info("Memory Statistics for 'customer-support':");
                    LOG.info("  Total Memories: {}", stats.getTotalMemories());
                    LOG.info("  Episodic: {}", stats.getEpisodicCount());
                    LOG.info("  Semantic: {}", stats.getSemanticCount());
                    LOG.info("  Procedural: {}", stats.getProceduralCount());
                    LOG.info("  Working: {}", stats.getWorkingCount());
                    LOG.info("  Avg Importance: {:.3f}", stats.getAvgImportance());
                    LOG.info("  Oldest Memory: {}", stats.getOldestMemory());
                    LOG.info("  Newest Memory: {}", stats.getNewestMemory());
                })
                .replaceWithVoid();
    }

    // ==================== HELPER METHODS ====================

    private Uni<String> storeCustomerIssue(String content) {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        return embeddingService.embed(content)
                .flatMap(embedding -> {
                    Memory memory = Memory.builder()
                            .namespace("customer-support")
                            .content(content)
                            .embedding(embedding)
                            .type(MemoryType.EPISODIC)
                            .importance(0.6)
                            .addMetadata("category", "issue")
                            .build();

                    return memoryStore.store(memory);
                });
    }

    private Uni<String> storeKnowledgeArticle(String content) {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        return embeddingService.embed(content)
                .flatMap(embedding -> {
                    Memory memory = Memory.builder()
                            .namespace("customer-support")
                            .content(content)
                            .embedding(embedding)
                            .type(MemoryType.SEMANTIC)
                            .importance(0.9)
                            .addMetadata("category", "knowledge")
                            .build();

                    return memoryStore.store(memory);
                });
    }

    private Uni<String> createTimedMemory(String content, Instant timestamp, double importance) {
        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        return embeddingService.embed(content)
                .flatMap(embedding -> {
                    Memory memory = Memory.builder()
                            .namespace("customer-support")
                            .content(content)
                            .embedding(embedding)
                            .type(MemoryType.EPISODIC)
                            .timestamp(timestamp)
                            .importance(importance)
                            .build();

                    return memoryStore.store(memory);
                });
    }
}
