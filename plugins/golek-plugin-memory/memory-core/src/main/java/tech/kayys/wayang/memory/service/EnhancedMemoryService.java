package tech.kayys.wayang.memory.service;

import tech.kayys.wayang.memory.context.*;
import tech.kayys.wayang.memory.model.ConversationMemory;
import tech.kayys.wayang.memory.model.MemoryContext;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.*;

/**
 * Enhanced Memory Service with advanced context engineering
 */
@ApplicationScoped
public class EnhancedMemoryService {

    private static final Logger LOG = LoggerFactory.getLogger(EnhancedMemoryService.class);

    @Inject
    MemoryService baseMemoryService;

    @Inject
    HierarchicalMemoryManager hierarchicalMemory;

    @Inject
    ContextWindowManager contextWindowManager;

    @Inject
    ContextCompressionService compressionService;

    @Inject
    RAGContextRetriever ragRetriever;

    @Inject
    MemoryIndexService indexService;

    @Inject
    MemoryReinforcementService reinforcementService;

    /**
     * Get enriched context with hierarchical memory structure
     */
    public Uni<EnrichedContext> getEnrichedContext(
            String sessionId,
            String userId,
            String currentTask) {

        LOG.info("Getting enriched context for session: {}, task: {}", sessionId, currentTask);

        return baseMemoryService.getContext(sessionId, userId)
            .onItem().transformToUni(context ->
                buildEnrichedContext(context, currentTask, userId)
            );
    }

    /**
     * Build enriched context with all advanced features
     */
    private Uni<EnrichedContext> buildEnrichedContext(
            MemoryContext baseContext,
            String currentTask,
            String userId) {

        return Uni.combine().all().unis(
            // Build hierarchical memory layers
            hierarchicalMemory.createEpisodicMemory(baseContext),
            hierarchicalMemory.extractSemanticMemory(baseContext.getConversations()),
            getUserProceduralMemory(userId),

            // Build optimal context window
            buildOptimalContextWindow(baseContext, currentTask),

            // Get relevant memories using RAG
            ragRetriever.hybridRetrieval(baseContext.getSessionId(), currentTask, 10)
        ).asTuple()
        .onItem().transformToUni(tuple -> {
            EpisodicMemory episodic = tuple.getItem1();
            SemanticMemory semantic = tuple.getItem2();
            ProceduralMemory procedural = tuple.getItem3();
            ContextWindow contextWindow = tuple.getItem4();
            List<ConversationMemory> relevantMemories = tuple.getItem5();

            // Build working memory
            return hierarchicalMemory.buildWorkingMemory(
                episodic, semantic, procedural, currentTask
            ).onItem().transform(workingMemory ->
                new EnrichedContext(
                    baseContext,
                    episodic,
                    semantic,
                    procedural,
                    workingMemory,
                    contextWindow,
                    relevantMemories,
                    Instant.now()
                )
            );
        });
    }

    /**
     * Store context with automatic indexing and compression
     */
    public Uni<Void> storeEnhancedContext(MemoryContext context) {
        return baseMemoryService.storeContext(context)
            .onItem().transformToUni(unused ->
                // Build indexes asynchronously
                indexService.buildInvertedIndex(
                    context.getSessionId(),
                    context.getConversations()
                )
            )
            .onItem().transformToUni(unused ->
                indexService.buildSemanticHashIndex(
                    context.getSessionId(),
                    context.getConversations()
                )
            )
            .onItem().transformToUni(unused ->
                // Check if compression needed
                checkAndCompress(context)
            );
    }

    /**
     * Retrieve context with automatic reinforcement
     */
    public Uni<MemoryContext> retrieveWithReinforcement(
            String sessionId,
            String userId,
            Map<String, List<Instant>> accessHistory) {

        return baseMemoryService.getContext(sessionId, userId)
            .onItem().transformToUni(context -> {
                // Calculate memory strengths
                return reinforcementService.calculateMemoryStrengths(
                    sessionId,
                    context.getConversations(),
                    accessHistory
                ).onItem().transformToUni(strengths -> {
                    // Identify and reinforce weak memories
                    return reinforcementService.identifyMemoriesForReinforcement(
                        context.getConversations(),
                        strengths
                    ).onItem().transformToUni(toReinforce -> {
                        // Apply reinforcement
                        List<Uni<Void>> reinforcements = toReinforce.stream()
                            .map(m -> reinforcementService.reinforceMemory(
                                m.getId(), accessHistory))
                            .collect(java.util.stream.Collectors.toList());

                        return Uni.combine().all().unis(reinforcements)
                            .discardItems()
                            .replaceWith(context);
                    });
                });
            });
    }

    /**
     * Smart context compression when needed
     */
    private Uni<Void> checkAndCompress(MemoryContext context) {
        if (context.getConversations().size() > 50) {
            LOG.info("Context size exceeds threshold, applying compression");

            return compressionService.compressContext(
                context.getConversations(),
                CompressionStrategy.HIERARCHICAL_CLUSTERING,
                0.3 // 30% compression
            ).onItem().transformToUni(compressed -> {
                // Store compressed version
                LOG.info("Compressed {} memories to {} units with {}% information retention",
                        compressed.getOriginalMemoryCount(),
                        compressed.getCompressedUnitCount(),
                        compressed.getInformationRetention() * 100);

                return Uni.createFrom().voidItem();
            });
        }

        return Uni.createFrom().voidItem();
    }

    /**
     * Build optimal context window for task
     */
    private Uni<ContextWindow> buildOptimalContextWindow(
            MemoryContext context,
            String currentTask) {

        Map<String, Object> constraints = Map.of(
            "maxTokens", 8000,
            "taskComplexity", analyzeTaskComplexity(currentTask)
        );

        return contextWindowManager.buildContextWindow(
            context.getConversations(),
            currentTask,
            constraints
        );
    }

    /**
     * Get user's procedural memory
     */
    private Uni<ProceduralMemory> getUserProceduralMemory(String userId) {
        // This would typically load historical contexts for the user
        return hierarchicalMemory.analyzeProceduralPatterns(
            userId,
            new ArrayList<>() // Placeholder
        );
    }

    private double analyzeTaskComplexity(String task) {
        return task.length() > 100 ? 0.8 : 0.5;
    }
}