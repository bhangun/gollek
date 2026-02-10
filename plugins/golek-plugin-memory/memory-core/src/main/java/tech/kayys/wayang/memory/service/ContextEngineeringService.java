package tech.kayys.gamelan.executor.memory;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Context engineering service for building optimized context from memories and
 * current query
 */
@ApplicationScoped
public class ContextEngineeringService {

    private static final Logger LOG = LoggerFactory.getLogger(ContextEngineeringService.class);

    @ConfigProperty(name = "gamelan.context.max-tokens", defaultValue = "8000")
    int maxContextTokens;

    @ConfigProperty(name = "gamelan.context.decay-rate", defaultValue = "0.001")
    double temporalDecayRate;

    @ConfigProperty(name = "gamelan.context.recency-weight", defaultValue = "0.3")
    double recencyWeight;

    @ConfigProperty(name = "gamelan.context.importance-weight", defaultValue = "0.4")
    double importanceWeight;

    @ConfigProperty(name = "gamelan.context.similarity-weight", defaultValue = "0.3")
    double similarityWeight;

    @Inject
    VectorMemoryStore memoryStore;

    @Inject
    EmbeddingServiceFactory embeddingFactory;

    @Inject
    TextChunker textChunker;

    /**
     * Build optimized context from memories and current query
     *
     * @param query     Current query/instruction
     * @param namespace Memory namespace
     * @param config    Context configuration
     * @return Optimized context object
     */
    public Uni<EngineerContext> buildContext(
            String query,
            String namespace,
            ContextConfig config) {

        LOG.info("Building context for query in namespace: {}", namespace);

        EmbeddingService embeddingService = embeddingFactory.getEmbeddingService();

        // Generate query embedding
        return embeddingService.embed(query)
                .flatMap(queryEmbedding -> retrieveRelevantMemories(
                        queryEmbedding,
                        namespace,
                        config))
                .map(memories -> assembleContext(query, memories, config));
    }

    /**
     * Retrieve relevant memories with multi-factor scoring
     *
     * @param queryEmbedding Query vector embedding
     * @param namespace      Memory namespace
     * @param config         Context configuration
     * @return List of relevant memories
     */
    private Uni<List<ScoredMemory>> retrieveRelevantMemories(
            float[] queryEmbedding,
            String namespace,
            ContextConfig config) {

        LOG.debug("Retrieving relevant memories with multi-factor scoring");

        // Search for semantically similar memories
        Map<String, Object> filters = new HashMap<>();
        filters.put("namespace", namespace);

        // Add type filters if specified
        if (config.getMemoryTypes() != null && !config.getMemoryTypes().isEmpty()) {
            filters.put("types", config.getMemoryTypes());
        }

        return memoryStore.search(
                queryEmbedding,
                config.getMaxMemories() * 2, // Retrieve more, then re-rank
                0.5, // Minimum similarity
                filters)
                .map(scoredMemories -> reRankWithMultipleFactors(scoredMemories, queryEmbedding))
                .map(reranked -> reranked.stream()
                        .limit(config.getMaxMemories())
                        .collect(Collectors.toList()));
    }

    /**
     * Re-rank memories using multiple factors:
     * - Semantic similarity
     * - Temporal recency
     * - Importance score
     * - Access frequency
     */
    private List<ScoredMemory> reRankWithMultipleFactors(
            List<ScoredMemory> memories,
            float[] queryEmbedding) {

        LOG.debug("Re-ranking {} memories with multiple factors", memories.size());

        List<ScoredMemory> reranked = new ArrayList<>();
        Instant now = Instant.now();

        for (ScoredMemory scoredMemory : memories) {
            Memory memory = scoredMemory.getMemory();

            // Original similarity score
            double similarityScore = scoredMemory.getScore();

            // Recency score (exponential decay)
            long ageMinutes = Duration.between(memory.getTimestamp(), now).toMinutes();
            double recencyScore = Math.exp(-temporalDecayRate * ageMinutes);

            // Importance score (from memory)
            double importanceScore = memory.getImportance();

            // Access frequency (from metadata)
            int accessCount = (int) memory.getMetadata().getOrDefault("accessCount", 0);
            double frequencyScore = Math.log(accessCount + 1) / Math.log(100); // Normalized log

            // Combined score
            double combinedScore = (similarityScore * similarityWeight) +
                    (recencyScore * recencyWeight) +
                    (importanceScore * importanceWeight) +
                    (frequencyScore * 0.1); // Small weight for frequency

            Map<String, Object> scoreBreakdown = Map.of(
                    "total", combinedScore,
                    "similarity", similarityScore,
                    "recency", recencyScore,
                    "importance", importanceScore,
                    "frequency", frequencyScore);

            reranked.add(new ScoredMemory(memory, combinedScore, scoreBreakdown));
        }

        // Sort by combined score
        reranked.sort((a, b) -> Double.compare(b.getScore(), a.getScore()));

        LOG.debug("Re-ranking complete, top score: {}",
                reranked.isEmpty() ? 0 : reranked.get(0).getScore());

        return reranked;
    }

    /**
     * Assemble context from memories with token budget optimization
     *
     * @param query    Original query
     * @param memories Relevant memories
     * @param config   Context configuration
     * @return Assembled context
     */
    private EngineerContext assembleContext(
            String query,
            List<ScoredMemory> memories,
            ContextConfig config) {

        LOG.debug("Assembling context from {} memories", memories.size());

        // Calculate token budget
        int queryTokens = textChunker.estimateTokenCount(query);
        int availableTokens = maxContextTokens - queryTokens - config.getReservedTokens();

        // Build context sections
        List<ContextSection> sections = new ArrayList<>();
        int usedTokens = 0;

        // 1. System instructions (if provided)
        if (config.getSystemPrompt() != null) {
            int systemTokens = textChunker.estimateTokenCount(config.getSystemPrompt());
            if (usedTokens + systemTokens <= availableTokens) {
                sections.add(new ContextSection(
                        "system",
                        config.getSystemPrompt(),
                        systemTokens,
                        1.0));
                usedTokens += systemTokens;
            }
        }

        // 2. Recent conversation history (if provided)
        if (config.getConversationHistory() != null) {
            int historyTokens = textChunker.estimateTokenCount(
                    String.join("\n", config.getConversationHistory()));
            if (usedTokens + historyTokens <= availableTokens * 0.3) { // Max 30% for history
                sections.add(new ContextSection(
                        "conversation_history",
                        String.join("\n", config.getConversationHistory()),
                        historyTokens,
                        0.9));
                usedTokens += historyTokens;
            }
        }

        // 3. Relevant memories
        for (ScoredMemory scoredMemory : memories) {
            Memory memory = scoredMemory.getMemory();
            String content = formatMemory(memory, config);
            int memoryTokens = textChunker.estimateTokenCount(content);

            if (usedTokens + memoryTokens <= availableTokens) {
                sections.add(new ContextSection(
                        "memory_" + memory.getType().name().toLowerCase(),
                        content,
                        memoryTokens,
                        scoredMemory.getScore()));
                usedTokens += memoryTokens;

                // Update access count
                Map<String, Object> metadata = new HashMap<>(memory.getMetadata());
                int accessCount = (int) metadata.getOrDefault("accessCount", 0);
                metadata.put("accessCount", accessCount + 1);
                metadata.put("lastAccessed", Instant.now().toString());

                memoryStore.updateMetadata(memory.getId(), metadata)
                        .subscribe().with(
                                updated -> LOG.trace("Updated access count for memory: {}", memory.getId()),
                                error -> LOG.warn("Failed to update memory metadata", error));
            } else {
                LOG.debug("Token budget exhausted, skipping remaining memories");
                break;
            }
        }

        // 4. Task instructions (if provided)
        if (config.getTaskInstructions() != null) {
            int instructionTokens = textChunker.estimateTokenCount(config.getTaskInstructions());
            if (usedTokens + instructionTokens <= availableTokens) {
                sections.add(new ContextSection(
                        "task_instructions",
                        config.getTaskInstructions(),
                        instructionTokens,
                        1.0));
                usedTokens += instructionTokens;
            }
        }

        EngineerContext context = new EngineerContext(
                query,
                sections,
                usedTokens,
                maxContextTokens);

        LOG.info("Context assembled: {} sections, {} tokens used of {} available",
                sections.size(), usedTokens, availableTokens);

        return context;
    }

    /**
     * Format memory for inclusion in context
     *
     * @param memory Memory to format
     * @param config Context configuration
     * @return Formatted memory string
     */
    private String formatMemory(Memory memory, ContextConfig config) {
        StringBuilder formatted = new StringBuilder();

        // Add metadata if enabled
        if (config.isIncludeMetadata()) {
            formatted.append("[")
                    .append(memory.getType().name())
                    .append(" - ")
                    .append(formatTimestamp(memory.getTimestamp()))
                    .append(" - Importance: ")
                    .append(String.format("%.2f", memory.getImportance()))
                    .append("]\n");
        }

        // Add content
        formatted.append(memory.getContent());

        return formatted.toString();
    }

    /**
     * Format timestamp for human readability
     */
    private String formatTimestamp(Instant timestamp) {
        Duration age = Duration.between(timestamp, Instant.now());

        if (age.toDays() > 0) {
            return age.toDays() + " days ago";
        } else if (age.toHours() > 0) {
            return age.toHours() + " hours ago";
        } else if (age.toMinutes() > 0) {
            return age.toMinutes() + " minutes ago";
        } else {
            return "just now";
        }
    }
}