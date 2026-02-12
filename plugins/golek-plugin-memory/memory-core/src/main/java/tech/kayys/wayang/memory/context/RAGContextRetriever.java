package tech.kayys.wayang.memory.context;

import tech.kayys.wayang.memory.model.ConversationMemory;
import tech.kayys.wayang.memory.service.MemoryServiceImpl;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Implements RAG (Retrieval-Augmented Generation) for context retrieval
 * Combines dense retrieval, sparse retrieval, and reranking
 */
@ApplicationScoped
public class RAGContextRetriever {
    
    private static final Logger LOG = LoggerFactory.getLogger(RAGContextRetriever.class);
    
    @Inject
    MemoryServiceImpl memoryService;
    
    @Inject
    ContextWindowManager contextWindowManager;

    /**
     * Hybrid retrieval: Combine dense (vector) and sparse (keyword) search
     */
    public Uni<List<ConversationMemory>> hybridRetrieval(
            String sessionId,
            String query,
            int topK) {
        
        LOG.debug("Performing hybrid retrieval for query: {}", query);
        
        return Uni.combine().all()
            .unis(
                denseRetrieval(sessionId, query, topK * 2),
                sparseRetrieval(sessionId, query, topK * 2)
            )
            .asTuple()
            .onItem().transform(tuple -> {
                List<ConversationMemory> denseResults = tuple.getItem1();
                List<ConversationMemory> sparseResults = tuple.getItem2();
                
                // Reciprocal Rank Fusion
                return reciprocalRankFusion(
                    denseResults,
                    sparseResults,
                    topK
                );
            });
    }

    /**
     * Dense retrieval using vector similarity
     */
    private Uni<List<ConversationMemory>> denseRetrieval(
            String sessionId,
            String query,
            int topK) {
        
        return memoryService.findSimilarMemories(sessionId, query, topK);
    }

    /**
     * Sparse retrieval using BM25-like keyword matching
     */
    private Uni<List<ConversationMemory>> sparseRetrieval(
            String sessionId,
            String query,
            int topK) {
        
        return memoryService.getContext(sessionId, null)
            .onItem().transform(context -> {
                String[] queryTerms = query.toLowerCase().split("\\s+");
                
                List<ScoredMemory> scored = context.getConversations().stream()
                    .map(memory -> new ScoredMemory(
                        memory,
                        calculateBM25Score(memory.getContent(), queryTerms, context.getConversations())
                    ))
                    .sorted(Comparator.comparing(ScoredMemory::getScore).reversed())
                    .limit(topK)
                    .collect(Collectors.toList());
                
                return scored.stream()
                    .map(ScoredMemory::getMemory)
                    .collect(Collectors.toList());
            });
    }

    /**
     * Rerank results using cross-encoder or advanced scoring
     */
    public Uni<List<ConversationMemory>> rerank(
            List<ConversationMemory> candidates,
            String query,
            int topK) {
        
        return Uni.createFrom().item(() -> {
            List<ScoredMemory> scored = candidates.stream()
                .map(memory -> new ScoredMemory(
                    memory,
                    calculateRerankScore(memory, query)
                ))
                .sorted(Comparator.comparing(ScoredMemory::getScore).reversed())
                .limit(topK)
                .collect(Collectors.toList());
            
            return scored.stream()
                .map(ScoredMemory::getMemory)
                .collect(Collectors.toList());
        });
    }

    /**
     * Reciprocal Rank Fusion for combining multiple rankings
     */
    private List<ConversationMemory> reciprocalRankFusion(
            List<ConversationMemory> ranking1,
            List<ConversationMemory> ranking2,
            int topK) {
        
        Map<String, Double> fusedScores = new HashMap<>();
        double k = 60.0; // RRF constant
        
        // Score from first ranking
        for (int i = 0; i < ranking1.size(); i++) {
            String id = ranking1.get(i).getId();
            fusedScores.merge(id, 1.0 / (k + i + 1), Double::sum);
        }
        
        // Score from second ranking
        for (int i = 0; i < ranking2.size(); i++) {
            String id = ranking2.get(i).getId();
            fusedScores.merge(id, 1.0 / (k + i + 1), Double::sum);
        }
        
        // Combine and sort
        Map<String, ConversationMemory> memoryMap = new HashMap<>();
        ranking1.forEach(m -> memoryMap.put(m.getId(), m));
        ranking2.forEach(m -> memoryMap.put(m.getId(), m));
        
        return fusedScores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(topK)
            .map(e -> memoryMap.get(e.getKey()))
            .collect(Collectors.toList());
    }

    /**
     * BM25 scoring for keyword-based retrieval
     */
    private double calculateBM25Score(
            String document,
            String[] queryTerms,
            List<ConversationMemory> corpus) {
        
        double k1 = 1.5;
        double b = 0.75;
        double avgDocLength = corpus.stream()
            .mapToInt(m -> m.getContent().split("\\s+").length)
            .average()
            .orElse(100.0);
        
        int docLength = document.split("\\s+").length;
        double score = 0.0;
        
        for (String term : queryTerms) {
            int termFreq = countOccurrences(term, document);
            int docFreq = (int) corpus.stream()
                .filter(m -> m.getContent().toLowerCase().contains(term))
                .count();
            
            double idf = Math.log((corpus.size() - docFreq + 0.5) / (docFreq + 0.5) + 1.0);
            double tf = (termFreq * (k1 + 1.0)) / 
                       (termFreq + k1 * (1.0 - b + b * (docLength / avgDocLength)));
            
            score += idf * tf;
        }
        
        return score;
    }

    private int countOccurrences(String term, String document) {
        return document.toLowerCase().split(term.toLowerCase(), -1).length - 1;
    }

    /**
     * Advanced reranking score combining multiple signals
     */
    private double calculateRerankScore(ConversationMemory memory, String query) {
        double semanticScore = calculateSemanticSimilarity(memory.getContent(), query);
        double recencyScore = calculateRecency(memory);
        double lengthScore = calculateLengthFit(memory.getContent());
        double roleScore = memory.getRole().equals("assistant") ? 1.0 : 0.8;
        
        return (semanticScore * 0.5) + (recencyScore * 0.2) + 
               (lengthScore * 0.2) + (roleScore * 0.1);
    }

    private double calculateSemanticSimilarity(String text, String query) {
        // Simplified semantic similarity
        String[] textWords = text.toLowerCase().split("\\s+");
        String[] queryWords = query.toLowerCase().split("\\s+");
        
        Set<String> textSet = new HashSet<>(Arrays.asList(textWords));
        Set<String> querySet = new HashSet<>(Arrays.asList(queryWords));
        
        Set<String> intersection = new HashSet<>(textSet);
        intersection.retainAll(querySet);
        
        Set<String> union = new HashSet<>(textSet);
        union.addAll(querySet);
        
        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }

    private double calculateRecency(ConversationMemory memory) {
        long ageSeconds = java.time.Duration.between(
            memory.getTimestamp(),
            java.time.Instant.now()
        ).getSeconds();
        
        return Math.exp(-ageSeconds / 7200.0); // 2-hour half-life
    }

    private double calculateLengthFit(String content) {
        int length = content.length();
        
        // Prefer medium-length content (100-800 chars)
        if (length < 100) return 0.6;
        if (length < 800) return 1.0;
        if (length < 1500) return 0.8;
        return 0.5;
    }
}
