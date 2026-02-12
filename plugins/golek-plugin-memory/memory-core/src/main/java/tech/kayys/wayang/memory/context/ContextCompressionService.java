package tech.kayys.wayang.memory.context;

import tech.kayys.wayang.memory.model.ConversationMemory;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Advanced context compression techniques to maximize information density
 */
@ApplicationScoped
public class ContextCompressionService {
    
    private static final Logger LOG = LoggerFactory.getLogger(ContextCompressionService.class);
    
    @Inject
    HierarchicalMemoryManager memoryManager;

    /**
     * Compress context using multiple strategies
     */
    public Uni<CompressedContext> compressContext(
            List<ConversationMemory> memories,
            CompressionStrategy strategy,
            double targetCompressionRatio) {
        
        LOG.info("Compressing {} memories with strategy: {}, target ratio: {}",
                memories.size(), strategy, targetCompressionRatio);
        
        return Uni.createFrom().item(() -> {
            switch (strategy) {
                case EXTRACTIVE_SUMMARIZATION:
                    return extractiveSummarization(memories, targetCompressionRatio);
                case ABSTRACTIVE_SUMMARIZATION:
                    return abstractiveSummarization(memories, targetCompressionRatio);
                case HIERARCHICAL_CLUSTERING:
                    return hierarchicalClustering(memories, targetCompressionRatio);
                case INFORMATION_BOTTLENECK:
                    return informationBottleneck(memories, targetCompressionRatio);
                default:
                    return extractiveSummarization(memories, targetCompressionRatio);
            }
        });
    }

    /**
     * Extractive: Select most important sentences
     */
    private CompressedContext extractiveSummarization(
            List<ConversationMemory> memories,
            double targetRatio) {
        
        List<ScoredSentence> allSentences = new ArrayList<>();
        
        for (ConversationMemory memory : memories) {
            String[] sentences = memory.getContent().split("[.!?]+");
            
            for (String sentence : sentences) {
                if (sentence.trim().length() > 10) {
                    double score = calculateSentenceImportance(sentence, memories);
                    allSentences.add(new ScoredSentence(
                        sentence.trim(),
                        score,
                        memory.getId()
                    ));
                }
            }
        }
        
        // Sort by importance and select top sentences
        allSentences.sort(Comparator.comparing(ScoredSentence::getScore).reversed());
        
        int targetLength = (int) (getTotalLength(memories) * targetRatio);
        List<String> selectedSentences = new ArrayList<>();
        int currentLength = 0;
        
        for (ScoredSentence scored : allSentences) {
            if (currentLength + scored.getSentence().length() <= targetLength) {
                selectedSentences.add(scored.getSentence());
                currentLength += scored.getSentence().length();
            }
        }
        
        String compressed = String.join(". ", selectedSentences) + ".";
        
        return new CompressedContext(
            compressed,
            memories.size(),
            selectedSentences.size(),
            calculateCompressionRatio(getTotalLength(memories), compressed.length()),
            CompressionStrategy.EXTRACTIVE_SUMMARIZATION,
            calculateInformationRetention(memories, compressed)
        );
    }

    /**
     * Abstractive: Generate new condensed representation
     */
    private CompressedContext abstractiveSummarization(
            List<ConversationMemory> memories,
            double targetRatio) {
        
        // Group by topic/theme
        Map<String, List<ConversationMemory>> topicGroups = groupByTopic(memories);
        
        StringBuilder summary = new StringBuilder();
        
        for (Map.Entry<String, List<ConversationMemory>> entry : topicGroups.entrySet()) {
            String topicSummary = generateTopicSummary(entry.getKey(), entry.getValue());
            summary.append(topicSummary).append(" ");
        }
        
        String compressed = summary.toString().trim();
        
        return new CompressedContext(
            compressed,
            memories.size(),
            topicGroups.size(),
            calculateCompressionRatio(getTotalLength(memories), compressed.length()),
            CompressionStrategy.ABSTRACTIVE_SUMMARIZATION,
            calculateInformationRetention(memories, compressed)
        );
    }

    /**
     * Hierarchical Clustering: Group similar memories and summarize
     */
    private CompressedContext hierarchicalClustering(
            List<ConversationMemory> memories,
            double targetRatio) {
        
        List<MemoryCluster> clusters = clusterMemories(memories);
        
        StringBuilder compressed = new StringBuilder();
        
        for (MemoryCluster cluster : clusters) {
            String clusterSummary = summarizeCluster(cluster);
            compressed.append(clusterSummary).append(" ");
        }
        
        String result = compressed.toString().trim();
        
        return new CompressedContext(
            result,
            memories.size(),
            clusters.size(),
            calculateCompressionRatio(getTotalLength(memories), result.length()),
            CompressionStrategy.HIERARCHICAL_CLUSTERING,
            calculateInformationRetention(memories, result)
        );
    }

    /**
     * Information Bottleneck: Preserve maximum mutual information
     */
    private CompressedContext informationBottleneck(
            List<ConversationMemory> memories,
            double targetRatio) {
        
        // Calculate information content of each memory
        List<MemoryInformation> infoList = memories.stream()
            .map(m -> new MemoryInformation(
                m,
                calculateInformationContent(m, memories),
                calculateRedundancy(m, memories)
            ))
            .sorted(Comparator.comparing(MemoryInformation::getNetInformation).reversed())
            .collect(Collectors.toList());
        
        int targetLength = (int) (getTotalLength(memories) * targetRatio);
        StringBuilder compressed = new StringBuilder();
        int currentLength = 0;
        
        for (MemoryInformation info : infoList) {
            String content = info.getMemory().getContent();
            if (currentLength + content.length() <= targetLength) {
                compressed.append(content).append(" ");
                currentLength += content.length();
            }
        }
        
        String result = compressed.toString().trim();
        
        return new CompressedContext(
            result,
            memories.size(),
            infoList.size(),
            calculateCompressionRatio(getTotalLength(memories), result.length()),
            CompressionStrategy.INFORMATION_BOTTLENECK,
            calculateInformationRetention(memories, result)
        );
    }

    // Helper methods
    private double calculateSentenceImportance(String sentence, List<ConversationMemory> context) {
        // TF-IDF-like scoring
        String[] words = sentence.toLowerCase().split("\\s+");
        double score = 0.0;
        
        for (String word : words) {
            if (word.length() > 3) {
                int termFreq = countOccurrences(word, sentence);
                int docFreq = countDocuments(word, context);
                
                double tf = Math.log(1 + termFreq);
                double idf = Math.log((double) context.size() / (1 + docFreq));
                
                score += tf * idf;
            }
        }
        
        return score / Math.max(1, words.length);
    }

    private int countOccurrences(String word, String text) {
        return text.toLowerCase().split(word.toLowerCase(), -1).length - 1;
    }

    private int countDocuments(String word, List<ConversationMemory> memories) {
        return (int) memories.stream()
            .filter(m -> m.getContent().toLowerCase().contains(word.toLowerCase()))
            .count();
    }

    private Map<String, List<ConversationMemory>> groupByTopic(List<ConversationMemory> memories) {
        Map<String, List<ConversationMemory>> groups = new HashMap<>();
        
        for (ConversationMemory memory : memories) {
            String topic = extractMainTopic(memory.getContent());
            groups.computeIfAbsent(topic, k -> new ArrayList<>()).add(memory);
        }
        
        return groups;
    }

    private String extractMainTopic(String content) {
        // Simple keyword extraction
        String[] words = content.toLowerCase().split("\\s+");
        Map<String, Integer> frequency = new HashMap<>();
        
        for (String word : words) {
            if (word.length() > 4 && !isStopWord(word)) {
                frequency.merge(word, 1, Integer::sum);
            }
        }
        
        return frequency.entrySet().stream()
            .max(Comparator.comparing(Map.Entry::getValue))
            .map(Map.Entry::getKey)
            .orElse("general");
    }

    private boolean isStopWord(String word) {
        Set<String> stopWords = Set.of(
            "the", "and", "for", "that", "this", "with", "from", "have",
            "they", "what", "when", "where", "which", "who", "will", "would"
        );
        return stopWords.contains(word.toLowerCase());
    }

    private String generateTopicSummary(String topic, List<ConversationMemory> memories) {
        String combined = memories.stream()
            .map(ConversationMemory::getContent)
            .collect(Collectors.joining(" "));
        
        // Extract key points
        String[] sentences = combined.split("[.!?]+");
        String mostRelevant = Arrays.stream(sentences)
            .max(Comparator.comparing(s -> countOccurrences(topic, s)))
            .orElse(sentences.length > 0 ? sentences[0] : "");
        
        return String.format("Regarding %s: %s", topic, mostRelevant.trim());
    }

    private List<MemoryCluster> clusterMemories(List<ConversationMemory> memories) {
        // Simple clustering based on content similarity
        List<MemoryCluster> clusters = new ArrayList<>();
        Set<String> processed = new HashSet<>();
        
        for (ConversationMemory memory : memories) {
            if (processed.contains(memory.getId())) continue;
            
            MemoryCluster cluster = new MemoryCluster();
            cluster.addMemory(memory);
            processed.add(memory.getId());
            
            // Find similar memories
            for (ConversationMemory other : memories) {
                if (!processed.contains(other.getId())) {
                    double similarity = calculateContentSimilarity(
                        memory.getContent(),
                        other.getContent()
                    );
                    
                    if (similarity > 0.5) {
                        cluster.addMemory(other);
                        processed.add(other.getId());
                    }
                }
            }
            
            clusters.add(cluster);
        }
        
        return clusters;
    }

    private double calculateContentSimilarity(String text1, String text2) {
        Set<String> words1 = new HashSet<>(Arrays.asList(text1.toLowerCase().split("\\s+")));
        Set<String> words2 = new HashSet<>(Arrays.asList(text2.toLowerCase().split("\\s+")));
        
        Set<String> intersection = new HashSet<>(words1);
        intersection.retainAll(words2);
        
        Set<String> union = new HashSet<>(words1);
        union.addAll(words2);
        
        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }

    private String summarizeCluster(MemoryCluster cluster) {
        List<ConversationMemory> memories = cluster.getMemories();
        
        // Find the most representative memory
        ConversationMemory representative = memories.stream()
            .max(Comparator.comparing(m -> 
                calculateAverageSimilarity(m, memories)))
            .orElse(memories.get(0));
        
        return representative.getContent();
    }

    private double calculateAverageSimilarity(
            ConversationMemory memory,
            List<ConversationMemory> cluster) {
        
        return cluster.stream()
            .filter(m -> !m.getId().equals(memory.getId()))
            .mapToDouble(m -> calculateContentSimilarity(
                memory.getContent(),
                m.getContent()
            ))
            .average()
            .orElse(0.0);
    }

    private double calculateInformationContent(
            ConversationMemory memory,
            List<ConversationMemory> context) {
        
        String[] words = memory.getContent().toLowerCase().split("\\s+");
        double entropy = 0.0;
        
        for (String word : words) {
            int freq = countDocuments(word, context);
            double prob = (double) freq / context.size();
            
            if (prob > 0) {
                entropy -= prob * Math.log(prob);
            }
        }
        
        return entropy;
    }

    private double calculateRedundancy(
            ConversationMemory memory,
            List<ConversationMemory> context) {
        
        return context.stream()
            .filter(m -> !m.getId().equals(memory.getId()))
            .mapToDouble(m -> calculateContentSimilarity(
                memory.getContent(),
                m.getContent()
            ))
            .max()
            .orElse(0.0);
    }

    private int getTotalLength(List<ConversationMemory> memories) {
        return memories.stream()
            .mapToInt(m -> m.getContent().length())
            .sum();
    }

    private double calculateCompressionRatio(int originalLength, int compressedLength) {
        return originalLength > 0 ? (double) compressedLength / originalLength : 1.0;
    }

    private double calculateInformationRetention(
            List<ConversationMemory> original,
            String compressed) {
        
        // Calculate keyword overlap
        Set<String> originalKeywords = original.stream()
            .flatMap(m -> Arrays.stream(m.getContent().toLowerCase().split("\\s+")))
            .filter(word -> word.length() > 4 && !isStopWord(word))
            .collect(Collectors.toSet());
        
        Set<String> compressedKeywords = Arrays.stream(compressed.toLowerCase().split("\\s+"))
            .filter(word -> word.length() > 4 && !isStopWord(word))
            .collect(Collectors.toSet());
        
        Set<String> intersection = new HashSet<>(originalKeywords);
        intersection.retainAll(compressedKeywords);
        
        return originalKeywords.isEmpty() ? 0.0 : 
               (double) intersection.size() / originalKeywords.size();
    }

    // Supporting classes
    private static class ScoredSentence {
        private final String sentence;
        private final double score;
        private final String sourceId;

        public ScoredSentence(String sentence, double score, String sourceId) {
            this.sentence = sentence;
            this.score = score;
            this.sourceId = sourceId;
        }

        public String getSentence() { return sentence; }
        public double getScore() { return score; }
        public String getSourceId() { return sourceId; }
    }

    private static class MemoryCluster {
        private final List<ConversationMemory> memories = new ArrayList<>();

        public void addMemory(ConversationMemory memory) {
            memories.add(memory);
        }

        public List<ConversationMemory> getMemories() {
            return memories;
        }
    }

    private static class MemoryInformation {
        private final ConversationMemory memory;
        private final double informationContent;
        private final double redundancy;

        public MemoryInformation(ConversationMemory memory, double informationContent, double redundancy) {
            this.memory = memory;
            this.informationContent = informationContent;
            this.redundancy = redundancy;
        }

        public ConversationMemory getMemory() { return memory; }
        public double getNetInformation() { return informationContent - redundancy; }
    }
}
