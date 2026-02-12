package tech.kayys.wayang.memory.context;

import tech.kayys.wayang.memory.model.ConversationMemory;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import io.vertx.mutiny.redis.client.RedisAPI;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Advanced indexing strategies for fast memory retrieval
 */
@ApplicationScoped
public class MemoryIndexService {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryIndexService.class);
    
    @Inject
    RedisAPI redisAPI;
    
    // In-memory inverted index for fast keyword search
    private final Map<String, Set<String>> invertedIndex = new ConcurrentHashMap<>();
    
    // Semantic hash index for approximate nearest neighbor search
    private final Map<String, Set<String>> semanticHashIndex = new ConcurrentHashMap<>();

    /**
     * Build inverted index for keyword search
     */
    public Uni<Void> buildInvertedIndex(String sessionId, List<ConversationMemory> memories) {
        LOG.debug("Building inverted index for session: {} with {} memories", 
                 sessionId, memories.size());
        
        return Uni.createFrom().item(() -> {
            Map<String, Set<String>> localIndex = new HashMap<>();
            
            for (ConversationMemory memory : memories) {
                String[] tokens = tokenize(memory.getContent());
                
                for (String token : tokens) {
                    String indexKey = sessionId + ":" + token;
                    localIndex.computeIfAbsent(indexKey, k -> new HashSet<>())
                             .add(memory.getId());
                }
            }
            
            // Update the global index
            invertedIndex.putAll(localIndex);
            
            // Persist to Redis for durability
            return persistIndexToRedis(sessionId, localIndex);
        }).flatMap(uni -> uni).replaceWithVoid();
    }

    /**
     * Build semantic hash index using LSH (Locality-Sensitive Hashing)
     */
    public Uni<Void> buildSemanticHashIndex(String sessionId, List<ConversationMemory> memories) {
        LOG.debug("Building semantic hash index for session: {}", sessionId);
        
        return Uni.createFrom().item(() -> {
            Map<String, Set<String>> localIndex = new HashMap<>();
            
            for (ConversationMemory memory : memories) {
                if (memory.getEmbedding() != null && !memory.getEmbedding().isEmpty()) {
                    String[] hashes = computeLocalitySensitiveHash(memory.getEmbedding());
                    
                    for (String hash : hashes) {
                        String indexKey = sessionId + ":lsh:" + hash;
                        localIndex.computeIfAbsent(indexKey, k -> new HashSet<>())
                                 .add(memory.getId());
                    }
                }
            }
            
            semanticHashIndex.putAll(localIndex);
            return null;
        }).replaceWithVoid();
    }

    /**
     * Fast keyword-based search using inverted index
     */
    public Uni<Set<String>> searchByKeywords(String sessionId, String query) {
        return Uni.createFrom().item(() -> {
            String[] queryTokens = tokenize(query);
            Set<String> results = new HashSet<>();
            
            for (String token : queryTokens) {
                String indexKey = sessionId + ":" + token;
                Set<String> memoryIds = invertedIndex.get(indexKey);
                
                if (memoryIds != null) {
                    if (results.isEmpty()) {
                        results.addAll(memoryIds);
                    } else {
                        // AND operation for multi-term queries
                        results.retainAll(memoryIds);
                    }
                }
            }
            
            return results;
        });
    }

    /**
     * Fast approximate nearest neighbor search using LSH index
     */
    public Uni<Set<String>> searchBySemantic(String sessionId, List<Double> queryEmbedding) {
        return Uni.createFrom().item(() -> {
            String[] queryHashes = computeLocalitySensitiveHash(queryEmbedding);
            Set<String> candidates = new HashSet<>();
            
            for (String hash : queryHashes) {
                String indexKey = sessionId + ":lsh:" + hash;
                Set<String> memoryIds = semanticHashIndex.get(indexKey);
                
                if (memoryIds != null) {
                    candidates.addAll(memoryIds);
                }
            }
            
            return candidates;
        });
    }

    /**
     * Tokenize text for indexing
     */
    private String[] tokenize(String text) {
        return text.toLowerCase()
                  .replaceAll("[^a-z0-9\\s]", "")
                  .split("\\s+");
    }

    /**
     * Compute LSH using random hyperplanes
     */
    private String[] computeLocalitySensitiveHash(List<Double> embedding) {
        int numHashFunctions = 5;
        int numBits = 8;
        String[] hashes = new String[numHashFunctions];
        
        Random random = new Random(42); // Fixed seed for reproducibility
        
        for (int h = 0; h < numHashFunctions; h++) {
            StringBuilder hashBits = new StringBuilder();
            
            for (int b = 0; b < numBits; b++) {
                // Generate random hyperplane
                List<Double> hyperplane = new ArrayList<>();
                for (int i = 0; i < embedding.size(); i++) {
                    hyperplane.add(random.nextGaussian());
                }
                
                // Compute dot product
                double dotProduct = 0.0;
                for (int i = 0; i < embedding.size(); i++) {
                    dotProduct += embedding.get(i) * hyperplane.get(i);
                }
                
                // Hash bit is 1 if dot product is positive
                hashBits.append(dotProduct >= 0 ? "1" : "0");
            }
            
            hashes[h] = hashBits.toString();
        }
        
        return hashes;
    }

    /**
     * Persist index to Redis for durability
     */
    private Uni<Void> persistIndexToRedis(String sessionId, Map<String, Set<String>> index) {
        if (index.isEmpty()) return Uni.createFrom().voidItem();
        
        List<Uni<Void>> operations = new ArrayList<>();
        
        for (Map.Entry<String, Set<String>> entry : index.entrySet()) {
            String redisKey = "index:" + entry.getKey();
            String[] values = entry.getValue().toArray(new String[0]);
            
            List<String> args = new ArrayList<>();
            args.add(redisKey);
            Collections.addAll(args, values);
            
            Uni<Void> op = redisAPI.sadd(args)
                .replaceWithVoid();
            operations.add(op);
        }
        
        return Uni.combine().all().unis(operations).discardItems();
    }

    /**
     * Clear index for a session
     */
    public Uni<Void> clearIndex(String sessionId) {
        return Uni.createFrom().item(() -> {
            // Remove from in-memory index
            invertedIndex.keySet().removeIf(key -> key.startsWith(sessionId + ":"));
            semanticHashIndex.keySet().removeIf(key -> key.startsWith(sessionId + ":"));
            
            return null;
        }).replaceWithVoid();
    }
}