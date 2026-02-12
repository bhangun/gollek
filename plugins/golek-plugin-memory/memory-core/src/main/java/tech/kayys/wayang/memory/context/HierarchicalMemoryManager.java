package tech.kayys.wayang.memory.context;

import tech.kayys.wayang.memory.model.ConversationMemory;
import tech.kayys.wayang.memory.model.MemoryContext;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Implements hierarchical memory with episodic, semantic, and procedural layers
 * Similar to human memory systems for better context retention and retrieval
 */
@ApplicationScoped
public class HierarchicalMemoryManager {
    
    private static final Logger LOG = LoggerFactory.getLogger(HierarchicalMemoryManager.class);

    /**
     * Episodic Memory: Recent, detailed conversations
     * Short-term memory with high detail
     */
    public Uni<EpisodicMemory> createEpisodicMemory(MemoryContext context) {
        LOG.debug("Creating episodic memory for session: {}", context.getSessionId());
        
        return Uni.createFrom().item(() -> {
            List<ConversationMemory> recentMemories = context.getConversations().stream()
                .filter(m -> m.getTimestamp().isAfter(Instant.now().minusSeconds(3600)))
                .sorted(Comparator.comparing(ConversationMemory::getTimestamp).reversed())
                .limit(20)
                .collect(Collectors.toList());
            
            return new EpisodicMemory(
                context.getSessionId(),
                recentMemories,
                calculateContextCoherence(recentMemories),
                Instant.now()
            );
        });
    }

    /**
     * Semantic Memory: Extracted knowledge and facts
     * Long-term memory with consolidated information
     */
    public Uni<SemanticMemory> extractSemanticMemory(List<ConversationMemory> memories) {
        LOG.debug("Extracting semantic memory from {} conversations", memories.size());
        
        return Uni.createFrom().item(() -> {
            Map<String, List<String>> knowledgeGraph = buildKnowledgeGraph(memories);
            List<Fact> extractedFacts = extractFacts(memories);
            Map<String, Double> conceptStrength = calculateConceptStrength(knowledgeGraph);
            
            return new SemanticMemory(
                knowledgeGraph,
                extractedFacts,
                conceptStrength,
                Instant.now()
            );
        });
    }

    /**
     * Procedural Memory: Learned patterns and behaviors
     * Skill memory for task execution patterns
     */
    public Uni<ProceduralMemory> analyzeProceduralPatterns(String userId, List<MemoryContext> historicalContexts) {
        LOG.debug("Analyzing procedural patterns for user: {}", userId);
        
        return Uni.createFrom().item(() -> {
            List<TaskPattern> patterns = identifyTaskPatterns(historicalContexts);
            Map<String, List<String>> commonSequences = findCommonSequences(historicalContexts);
            Map<String, Double> skillProficiency = calculateSkillProficiency(patterns);
            
            return new ProceduralMemory(
                userId,
                patterns,
                commonSequences,
                skillProficiency,
                Instant.now()
            );
        });
    }

    /**
     * Working Memory: Active context window
     * Attention-focused current task context
     */
    public Uni<WorkingMemory> buildWorkingMemory(
            EpisodicMemory episodic,
            SemanticMemory semantic,
            ProceduralMemory procedural,
            String currentTask) {
        
        return Uni.createFrom().item(() -> {
            // Select most relevant memories based on current task
            List<ConversationMemory> relevantEpisodic = selectRelevantEpisodic(episodic, currentTask);
            List<Fact> relevantFacts = selectRelevantFacts(semantic, currentTask);
            List<TaskPattern> relevantPatterns = selectRelevantPatterns(procedural, currentTask);
            
            return new WorkingMemory(
                currentTask,
                relevantEpisodic,
                relevantFacts,
                relevantPatterns,
                calculateAttentionWeights(relevantEpisodic, relevantFacts),
                Instant.now()
            );
        });
    }

    // Helper methods
    private double calculateContextCoherence(List<ConversationMemory> memories) {
        if (memories.size() < 2) return 1.0;
        
        double totalSimilarity = 0.0;
        int comparisons = 0;
        
        for (int i = 0; i < memories.size() - 1; i++) {
            for (int j = i + 1; j < Math.min(i + 5, memories.size()); j++) {
                totalSimilarity += calculateSimilarity(
                    memories.get(i).getEmbedding(),
                    memories.get(j).getEmbedding()
                );
                comparisons++;
            }
        }
        
        return comparisons > 0 ? totalSimilarity / comparisons : 0.0;
    }

    private Map<String, List<String>> buildKnowledgeGraph(List<ConversationMemory> memories) {
        Map<String, List<String>> graph = new HashMap<>();
        
        for (ConversationMemory memory : memories) {
            List<String> entities = extractEntities(memory.getContent());
            List<String> relations = extractRelations(memory.getContent());
            
            for (String entity : entities) {
                graph.computeIfAbsent(entity, k -> new ArrayList<>()).addAll(relations);
            }
        }
        
        return graph;
    }

    private List<Fact> extractFacts(List<ConversationMemory> memories) {
        List<Fact> facts = new ArrayList<>();
        
        for (ConversationMemory memory : memories) {
            // Simple fact extraction - in production, use NLP
            if (memory.getContent().contains(" is ") || memory.getContent().contains(" are ")) {
                facts.add(new Fact(
                    extractSubject(memory.getContent()),
                    extractPredicate(memory.getContent()),
                    extractObject(memory.getContent()),
                    1.0,
                    memory.getTimestamp()
                ));
            }
        }
        
        return facts;
    }

    private Map<String, Double> calculateConceptStrength(Map<String, List<String>> knowledgeGraph) {
        Map<String, Double> strength = new HashMap<>();
        
        for (Map.Entry<String, List<String>> entry : knowledgeGraph.entrySet()) {
            // Strength based on connections and frequency
            double conceptStrength = Math.log(entry.getValue().size() + 1) * 0.5;
            strength.put(entry.getKey(), Math.min(conceptStrength, 1.0));
        }
        
        return strength;
    }

    private List<TaskPattern> identifyTaskPatterns(List<MemoryContext> contexts) {
        List<TaskPattern> patterns = new ArrayList<>();
        Map<String, Integer> taskFrequency = new HashMap<>();
        Map<String, List<String>> taskSequences = new HashMap<>();
        
        for (MemoryContext context : contexts) {
            String taskType = inferTaskType(context);
            taskFrequency.merge(taskType, 1, Integer::sum);
            
            List<String> steps = extractTaskSteps(context);
            taskSequences.computeIfAbsent(taskType, k -> new ArrayList<>()).addAll(steps);
        }
        
        for (Map.Entry<String, Integer> entry : taskFrequency.entrySet()) {
            patterns.add(new TaskPattern(
                entry.getKey(),
                entry.getValue(),
                findCommonSteps(taskSequences.get(entry.getKey())),
                calculateSuccessRate(entry.getKey(), contexts)
            ));
        }
        
        return patterns;
    }

    private Map<String, List<String>> findCommonSequences(List<MemoryContext> contexts) {
        Map<String, List<String>> sequences = new HashMap<>();
        
        for (MemoryContext context : contexts) {
            List<String> actions = extractActions(context);
            
            for (int i = 0; i < actions.size() - 1; i++) {
                String key = actions.get(i);
                String next = actions.get(i + 1);
                sequences.computeIfAbsent(key, k -> new ArrayList<>()).add(next);
            }
        }
        
        return sequences;
    }

    private Map<String, Double> calculateSkillProficiency(List<TaskPattern> patterns) {
        Map<String, Double> proficiency = new HashMap<>();
        
        for (TaskPattern pattern : patterns) {
            proficiency.put(
                pattern.getTaskType(),
                pattern.getSuccessRate() * Math.log(pattern.getFrequency() + 1) / 10.0
            );
        }
        
        return proficiency;
    }

    private List<ConversationMemory> selectRelevantEpisodic(EpisodicMemory episodic, String currentTask) {
        // Select memories relevant to current task using semantic similarity
        return episodic.getRecentMemories().stream()
            .filter(m -> isRelevantToTask(m.getContent(), currentTask))
            .limit(10)
            .collect(Collectors.toList());
    }

    private List<Fact> selectRelevantFacts(SemanticMemory semantic, String currentTask) {
        return semantic.getFacts().stream()
            .filter(f -> isFactRelevant(f, currentTask))
            .sorted(Comparator.comparing(Fact::getConfidence).reversed())
            .limit(5)
            .collect(Collectors.toList());
    }

    private List<TaskPattern> selectRelevantPatterns(ProceduralMemory procedural, String currentTask) {
        String taskType = inferTaskTypeFromDescription(currentTask);
        
        return procedural.getPatterns().stream()
            .filter(p -> p.getTaskType().equalsIgnoreCase(taskType))
            .sorted(Comparator.comparing(TaskPattern::getSuccessRate).reversed())
            .limit(3)
            .collect(Collectors.toList());
    }

    private Map<String, Double> calculateAttentionWeights(
            List<ConversationMemory> episodic,
            List<Fact> facts) {
        
        Map<String, Double> weights = new HashMap<>();
        
        // Recent memories get higher attention
        for (int i = 0; i < episodic.size(); i++) {
            double recencyWeight = 1.0 - (i * 0.1);
            weights.put("episodic_" + i, Math.max(recencyWeight, 0.1));
        }
        
        // High-confidence facts get higher attention
        for (int i = 0; i < facts.size(); i++) {
            weights.put("fact_" + i, facts.get(i).getConfidence());
        }
        
        return weights;
    }

    // Utility methods
    private double calculateSimilarity(List<Double> vec1, List<Double> vec2) {
        if (vec1 == null || vec2 == null || vec1.isEmpty() || vec2.isEmpty()) return 0.0;
        if (vec1.size() != vec2.size()) return 0.0;
        
        double dotProduct = 0.0, norm1 = 0.0, norm2 = 0.0;
        for (int i = 0; i < vec1.size(); i++) {
            dotProduct += vec1.get(i) * vec2.get(i);
            norm1 += vec1.get(i) * vec1.get(i);
            norm2 += vec2.get(i) * vec2.get(i);
        }
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    private List<String> extractEntities(String content) {
        // Simple entity extraction - in production use NER
        return Arrays.stream(content.split("\\s+"))
            .filter(word -> word.length() > 3 && Character.isUpperCase(word.charAt(0)))
            .collect(Collectors.toList());
    }

    private List<String> extractRelations(String content) {
        // Extract verbs as relations
        List<String> relations = new ArrayList<>();
        String[] words = content.split("\\s+");
        
        for (String word : words) {
            if (word.matches(".*ing$|.*ed$|.*es$")) {
                relations.add(word);
            }
        }
        
        return relations;
    }

    private String extractSubject(String content) {
        String[] parts = content.split(" is | are ");
        return parts.length > 0 ? parts[0].trim() : "";
    }

    private String extractPredicate(String content) {
        if (content.contains(" is ")) return "is";
        if (content.contains(" are ")) return "are";
        return "relates_to";
    }

    private String extractObject(String content) {
        String[] parts = content.split(" is | are ");
        return parts.length > 1 ? parts[1].trim() : "";
    }

    private String inferTaskType(MemoryContext context) {
        String combined = context.getConversations().stream()
            .map(ConversationMemory::getContent)
            .collect(Collectors.joining(" "));
        
        if (combined.toLowerCase().contains("code") || combined.toLowerCase().contains("program")) {
            return "CODING";
        } else if (combined.toLowerCase().contains("analyze") || combined.toLowerCase().contains("data")) {
            return "ANALYSIS";
        } else if (combined.toLowerCase().contains("write") || combined.toLowerCase().contains("document")) {
            return "WRITING";
        }
        return "GENERAL";
    }

    private List<String> extractTaskSteps(MemoryContext context) {
        return context.getConversations().stream()
            .filter(m -> m.getRole().equals("assistant"))
            .map(m -> summarizeStep(m.getContent()))
            .collect(Collectors.toList());
    }

    private String summarizeStep(String content) {
        // Simple step summarization
        return content.length() > 50 ? content.substring(0, 50) + "..." : content;
    }

    private List<String> findCommonSteps(List<String> steps) {
        Map<String, Integer> frequency = new HashMap<>();
        
        for (String step : steps) {
            frequency.merge(step, 1, Integer::sum);
        }
        
        return frequency.entrySet().stream()
            .filter(e -> e.getValue() > 1)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }

    private double calculateSuccessRate(String taskType, List<MemoryContext> contexts) {
        long successful = contexts.stream()
            .filter(c -> inferTaskType(c).equals(taskType))
            .filter(c -> c.getMetadata().getOrDefault("success", "false").equals("true"))
            .count();
        
        long total = contexts.stream()
            .filter(c -> inferTaskType(c).equals(taskType))
            .count();
        
        return total > 0 ? (double) successful / total : 0.5;
    }

    private List<String> extractActions(MemoryContext context) {
        return context.getConversations().stream()
            .map(m -> extractAction(m.getContent()))
            .filter(Objects::nonNull)
            .collect(Collectors.toList());
    }

    private String extractAction(String content) {
        // Extract verbs as actions
        String[] words = content.split("\\s+");
        for (String word : words) {
            if (word.matches("(run|execute|create|delete|update|read|write|analyze).*")) {
                return word;
            }
        }
        return null;
    }

    private boolean isRelevantToTask(String content, String task) {
        String[] taskWords = task.toLowerCase().split("\\s+");
        String lowerContent = content.toLowerCase();
        
        long matchCount = Arrays.stream(taskWords)
            .filter(lowerContent::contains)
            .count();
        
        return matchCount >= taskWords.length * 0.3;
    }

    private boolean isFactRelevant(Fact fact, String task) {
        String taskLower = task.toLowerCase();
        return taskLower.contains(fact.getSubject().toLowerCase()) ||
               taskLower.contains(fact.getObject().toLowerCase());
    }

    private String inferTaskTypeFromDescription(String description) {
        String lower = description.toLowerCase();
        if (lower.contains("code") || lower.contains("program")) return "CODING";
        if (lower.contains("analyze") || lower.contains("data")) return "ANALYSIS";
        if (lower.contains("write") || lower.contains("document")) return "WRITING";
        return "GENERAL";
    }
}
