package tech.kayys.wayang.memory.context;

import tech.kayys.wayang.memory.model.ConversationMemory;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Manages context window with intelligent token management
 * Implements strategies like sliding window, importance-based retention, etc.
 */
@ApplicationScoped
public class ContextWindowManager {
    
    private static final Logger LOG = LoggerFactory.getLogger(ContextWindowManager.class);
    
    @ConfigProperty(name = "context.window.max.tokens", defaultValue = "8000")
    int maxTokens;
    
    @ConfigProperty(name = "context.window.strategy", defaultValue = "IMPORTANCE_BASED")
    ContextWindowStrategy strategy;

    /**
     * Build optimized context window for current task
     */
    public Uni<ContextWindow> buildContextWindow(
            List<ConversationMemory> availableMemories,
            String currentTask,
            Map<String, Object> constraints) {
        
        LOG.debug("Building context window for task with {} available memories", availableMemories.size());
        
        return Uni.createFrom().item(() -> {
            switch (strategy) {
                case SLIDING_WINDOW:
                    return buildSlidingWindow(availableMemories);
                case IMPORTANCE_BASED:
                    return buildImportanceBasedWindow(availableMemories, currentTask);
                case HIERARCHICAL:
                    return buildHierarchicalWindow(availableMemories, currentTask);
                case ADAPTIVE:
                    return buildAdaptiveWindow(availableMemories, currentTask, constraints);
                default:
                    return buildSlidingWindow(availableMemories);
            }
        });
    }

    /**
     * Sliding Window: Most recent N memories
     */
    private ContextWindow buildSlidingWindow(List<ConversationMemory> memories) {
        List<ConversationMemory> sorted = memories.stream()
            .sorted(Comparator.comparing(ConversationMemory::getTimestamp).reversed())
            .collect(Collectors.toList());
        
        List<ConversationMemory> selected = new ArrayList<>();
        int tokenCount = 0;
        
        for (ConversationMemory memory : sorted) {
            int memoryTokens = estimateTokens(memory.getContent());
            if (tokenCount + memoryTokens <= maxTokens) {
                selected.add(memory);
                tokenCount += memoryTokens;
            } else {
                break;
            }
        }
        
        return new ContextWindow(
            selected,
            tokenCount,
            ContextWindowStrategy.SLIDING_WINDOW,
            calculateWindowQuality(selected)
        );
    }

    /**
     * Importance-Based: Select memories by relevance and importance
     */
    private ContextWindow buildImportanceBasedWindow(List<ConversationMemory> memories, String currentTask) {
        List<ScoredMemory> scored = memories.stream()
            .map(m -> new ScoredMemory(
                m,
                calculateImportanceScore(m, currentTask)
            ))
            .sorted(Comparator.comparing(ScoredMemory::getScore).reversed())
            .collect(Collectors.toList());
        
        List<ConversationMemory> selected = new ArrayList<>();
        int tokenCount = 0;
        
        for (ScoredMemory scoredMemory : scored) {
            int memoryTokens = estimateTokens(scoredMemory.getMemory().getContent());
            if (tokenCount + memoryTokens <= maxTokens) {
                selected.add(scoredMemory.getMemory());
                tokenCount += memoryTokens;
            }
        }
        
        // Re-sort by timestamp for coherent conversation flow
        selected.sort(Comparator.comparing(ConversationMemory::getTimestamp));
        
        return new ContextWindow(
            selected,
            tokenCount,
            ContextWindowStrategy.IMPORTANCE_BASED,
            calculateWindowQuality(selected)
        );
    }

    /**
     * Hierarchical: Combine summaries with detailed recent context
     */
    private ContextWindow buildHierarchicalWindow(List<ConversationMemory> memories, String currentTask) {
        List<ConversationMemory> summaries = memories.stream()
            .filter(m -> m.getMetadata().getOrDefault("type", "").equals("summary"))
            .collect(Collectors.toList());
        
        List<ConversationMemory> recent = memories.stream()
            .filter(m -> !m.getMetadata().getOrDefault("type", "").equals("summary"))
            .sorted(Comparator.comparing(ConversationMemory::getTimestamp).reversed())
            .limit(10)
            .collect(Collectors.toList());
        
        List<ConversationMemory> selected = new ArrayList<>();
        int tokenCount = 0;
        
        // Add summaries first (compressed historical context)
        for (ConversationMemory summary : summaries) {
            int tokens = estimateTokens(summary.getContent());
            if (tokenCount + tokens <= maxTokens * 0.3) { // Use 30% for summaries
                selected.add(summary);
                tokenCount += tokens;
            }
        }
        
        // Add recent detailed memories
        for (ConversationMemory memory : recent) {
            int tokens = estimateTokens(memory.getContent());
            if (tokenCount + tokens <= maxTokens) {
                selected.add(memory);
                tokenCount += tokens;
            }
        }
        
        return new ContextWindow(
            selected,
            tokenCount,
            ContextWindowStrategy.HIERARCHICAL,
            calculateWindowQuality(selected)
        );
    }

    /**
     * Adaptive: Dynamically adjust strategy based on task and constraints
     */
    private ContextWindow buildAdaptiveWindow(
            List<ConversationMemory> memories,
            String currentTask,
            Map<String, Object> constraints) {
        
        boolean needsHistory = analyzeHistoryNeed(currentTask);
        boolean isComplexTask = analyzeTaskComplexity(currentTask) > 0.7;
        int availableTokens = (int) constraints.getOrDefault("maxTokens", maxTokens);
        
        if (needsHistory && availableTokens > 6000) {
            return buildHierarchicalWindow(memories, currentTask);
        } else if (isComplexTask) {
            return buildImportanceBasedWindow(memories, currentTask);
        } else {
            return buildSlidingWindow(memories);
        }
    }

    /**
     * Calculate importance score for a memory relative to current task
     */
    private double calculateImportanceScore(ConversationMemory memory, String currentTask) {
        double recencyScore = calculateRecencyScore(memory);
        double relevanceScore = calculateRelevanceScore(memory, currentTask);
        double lengthScore = calculateLengthScore(memory);
        double roleScore = memory.getRole().equals("user") ? 0.8 : 1.0; // Assistant responses more important
        
        return (recencyScore * 0.3) + (relevanceScore * 0.5) + (lengthScore * 0.1) + (roleScore * 0.1);
    }

    private double calculateRecencyScore(ConversationMemory memory) {
        long ageSeconds = java.time.Duration.between(
            memory.getTimestamp(),
            java.time.Instant.now()
        ).getSeconds();
        
        // Exponential decay: half-life of 1 hour
        return Math.exp(-ageSeconds / 3600.0);
    }

    private double calculateRelevanceScore(ConversationMemory memory, String currentTask) {
        String[] taskWords = currentTask.toLowerCase().split("\\s+");
        String content = memory.getContent().toLowerCase();
        
        long matches = Arrays.stream(taskWords)
            .filter(content::contains)
            .count();
        
        return Math.min(1.0, (double) matches / taskWords.length * 2);
    }

    private double calculateLengthScore(ConversationMemory memory) {
        int length = memory.getContent().length();
        
        // Prefer medium-length memories (50-500 chars)
        if (length < 50) return 0.5;
        if (length < 500) return 1.0;
        if (length < 1000) return 0.8;
        return 0.6;
    }

    private int estimateTokens(String text) {
        // Rough estimation: 1 token ≈ 4 characters
        return text.length() / 4;
    }

    private boolean analyzeHistoryNeed(String task) {
        String lower = task.toLowerCase();
        return lower.contains("previous") || lower.contains("earlier") ||
               lower.contains("history") || lower.contains("before");
    }

    private double analyzeTaskComplexity(String task) {
        // Analyze based on length, technical terms, multi-step indicators
        int length = task.length();
        long technicalTerms = Arrays.stream(task.split("\\s+"))
            .filter(word -> word.length() > 8)
            .count();
        
        boolean multiStep = task.contains("then") || task.contains("after") ||
                           task.contains("first") || task.contains("finally");
        
        double complexity = Math.min(1.0, (length / 500.0) + (technicalTerms / 10.0));
        return multiStep ? Math.min(1.0, complexity + 0.3) : complexity;
    }

    private double calculateWindowQuality(List<ConversationMemory> memories) {
        if (memories.isEmpty()) return 0.0;
        
        // Quality metrics: diversity, coherence, coverage
        double diversity = calculateDiversity(memories);
        double coherence = calculateCoherence(memories);
        double coverage = Math.min(1.0, memories.size() / 20.0);
        
        return (diversity * 0.3) + (coherence * 0.4) + (coverage * 0.3);
    }

    private double calculateDiversity(List<ConversationMemory> memories) {
        Set<String> uniqueTopics = new HashSet<>();
        
        for (ConversationMemory memory : memories) {
            String[] words = memory.getContent().toLowerCase().split("\\s+");
            uniqueTopics.addAll(Arrays.asList(words).subList(0, Math.min(5, words.length)));
        }
        
        return Math.min(1.0, uniqueTopics.size() / 50.0);
    }

    private double calculateCoherence(List<ConversationMemory> memories) {
        // Temporal coherence - are memories in logical order?
        if (memories.size() < 2) return 1.0;
        
        int orderedPairs = 0;
        for (int i = 0; i < memories.size() - 1; i++) {
            if (memories.get(i).getTimestamp().isBefore(memories.get(i + 1).getTimestamp())) {
                orderedPairs++;
            }
        }
        
        return (double) orderedPairs / (memories.size() - 1);
    }

    private static class ScoredMemory {
        private final ConversationMemory memory;
        private final double score;

        public ScoredMemory(ConversationMemory memory, double score) {
            this.memory = memory;
            this.score = score;
        }

        public ConversationMemory getMemory() { return memory; }
        public double getScore() { return score; }
    }
}
