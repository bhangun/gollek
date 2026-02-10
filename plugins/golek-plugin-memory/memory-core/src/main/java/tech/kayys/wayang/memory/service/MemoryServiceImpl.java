package tech.kayys.wayang.memory.service;

import tech.kayys.wayang.memory.entity.*;
import tech.kayys.wayang.memory.model.*;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import io.quarkus.hibernate.reactive.panache.common.WithTransaction;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.vertx.mutiny.redis.client.RedisAPI;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

@ApplicationScoped
public class MemoryServiceImpl implements MemoryService {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryServiceImpl.class);
    
    @Inject
    EmbeddingModel embeddingModel;
    
    @Inject
    RedisAPI redisAPI;
    
    @Inject
    MemoryEventPublisher eventPublisher;
    
    @ConfigProperty(name = "memory.cache.ttl", defaultValue = "PT1H")
    Duration cacheTTL;
    
    @ConfigProperty(name = "memory.max.conversations", defaultValue = "50")
    int maxConversations;
    
    @ConfigProperty(name = "memory.similarity.threshold", defaultValue = "0.7")
    double similarityThreshold;

    @Override
    @WithTransaction
    public Uni<MemoryContext> getContext(String sessionId, String userId) {
        LOG.info("Retrieving memory context for session: {}, user: {}", sessionId, userId);
        
        String cacheKey = "memory:context:" + sessionId;
        
        return redisAPI.get(cacheKey)
            .onItem().transformToUni(cached -> {
                if (cached != null) {
                    return deserializeContext(cached.toString());
                } else {
                    return loadContextFromDatabase(sessionId, userId)
                        .onItem().transformToUni(context -> 
                            cacheContext(cacheKey, context)
                                .replaceWith(context)
                        );
                }
            })
            .onFailure().recoverWithUni(throwable -> {
                LOG.warn("Failed to retrieve from cache, loading from database", throwable);
                return loadContextFromDatabase(sessionId, userId);
            });
    }

    @Override
    @WithTransaction
    public Uni<Void> storeContext(MemoryContext context) {
        LOG.info("Storing memory context for session: {}", context.getSessionId());
        
        return persistContextToDatabase(context)
            .onItem().transformToUni(unused -> {
                String cacheKey = "memory:context:" + context.getSessionId();
                return cacheContext(cacheKey, context);
            })
            .onItem().transformToUni(unused -> 
                eventPublisher.publishMemoryUpdated(context)
            )
            .replaceWithVoid();
    }

    @Override
    @WithTransaction
    public Uni<Void> storeExecutionResult(String sessionId, AgentResponse result) {
        LOG.info("Storing execution result for session: {}, result: {}", sessionId, result.getId());
        
        return createConversationMemory(result)
            .onItem().transformToUni(memory -> 
                persistExecutionResult(result)
                    .onItem().transformToUni(unused -> 
                        addMemoryToSession(sessionId, memory)
                    )
            )
            .onItem().transformToUni(unused -> {
                String cacheKey = "memory:context:" + sessionId;
                return invalidateCache(cacheKey);
            })
            .onItem().transformToUni(unused -> 
                eventPublisher.publishExecutionStored(sessionId, result)
            )
            .replaceWithVoid();
    }

    @Override
    public Uni<List<AgentResponse>> getRecentResults(String sessionId, int limit) {
        LOG.info("Retrieving recent results for session: {}, limit: {}", sessionId, limit);
        
        return ExecutionResultEntity.<ExecutionResultEntity>find(
                "sessionId = ?1 ORDER BY timestamp DESC", sessionId)
            .page(0, limit)
            .list()
            .onItem().transform(entities -> 
                entities.stream()
                    .map(this::convertToAgentResponse)
                    .collect(Collectors.toList())
            );
    }

    // Additional semantic search capabilities
    public Uni<List<ConversationMemory>> findSimilarMemories(String sessionId, String query, int limit) {
        LOG.info("Finding similar memories for session: {}, query: {}", sessionId, query);
        
        return generateEmbedding(query)
            .onItem().transformToUni(queryEmbedding -> 
                ConversationMemoryEntity.<ConversationMemoryEntity>find("sessionId = ?1", sessionId)
                    .list()
                    .onItem().transform(entities -> 
                        entities.stream()
                            .map(entity -> {
                                double similarity = calculateCosineSimilarity(
                                    queryEmbedding, entity.embedding);
                                return convertToConversationMemory(entity, similarity);
                            })
                            .filter(memory -> memory.getRelevanceScore() > similarityThreshold)
                            .sorted((m1, m2) -> Double.compare(m2.getRelevanceScore(), m1.getRelevanceScore()))
                            .limit(limit)
                            .collect(Collectors.toList())
                    )
            );
    }

    public Uni<MemoryContext> summarizeAndCompact(String sessionId) {
        LOG.info("Summarizing and compacting memory for session: {}", sessionId);
        
        return getContext(sessionId, null)
            .onItem().transformToUni(context -> {
                if (context.getConversations().size() > maxConversations) {
                    List<ConversationMemory> toSummarize = context.getConversations()
                        .subList(0, context.getConversations().size() - maxConversations/2);
                    
                    return createSummaryMemory(toSummarize)
                        .onItem().transformToUni(summary -> {
                            List<ConversationMemory> compactedConversations = new ArrayList<>();
                            compactedConversations.add(summary);
                            compactedConversations.addAll(
                                context.getConversations().subList(maxConversations/2, context.getConversations().size())
                            );
                            
                            MemoryContext compactedContext = new MemoryContext(
                                context.getSessionId(),
                                context.getUserId(),
                                compactedConversations,
                                context.getMetadata(),
                                context.getCreatedAt(),
                                Instant.now()
                            );
                            
                            return storeContext(compactedContext)
                                .replaceWith(compactedContext);
                        });
                }
                return Uni.createFrom().item(context);
            });
    }

    // Private helper methods
    private Uni<MemoryContext> loadContextFromDatabase(String sessionId, String userId) {
        return MemorySessionEntity.<MemorySessionEntity>findById(sessionId)
            .onItem().transformToUni(entity -> {
                if (entity == null) {
                    return createNewSession(sessionId, userId);
                }
                
                return ConversationMemoryEntity.<ConversationMemoryEntity>find(
                        "sessionId = ?1 ORDER BY timestamp ASC", sessionId)
                    .list()
                    .onItem().transform(memories -> 
                        new MemoryContext(
                            entity.sessionId,
                            entity.userId,
                            memories.stream()
                                .map(m -> convertToConversationMemory(m, null))
                                .collect(Collectors.toList()),
                            entity.metadata != null ? new HashMap<>(entity.metadata) : new HashMap<>(),
                            entity.createdAt,
                            entity.updatedAt
                        )
                    );
            });
    }

    private Uni<MemoryContext> createNewSession(String sessionId, String userId) {
        MemorySessionEntity entity = new MemorySessionEntity();
        entity.sessionId = sessionId;
        entity.userId = userId;
        entity.metadata = new HashMap<>();
        entity.createdAt = Instant.now();
        entity.updatedAt = Instant.now();
        entity.expiresAt = Instant.now().plus(Duration.ofDays(30));
        
        return entity.persist()
            .onItem().transform(unused -> 
                new MemoryContext(
                    sessionId,
                    userId,
                    new ArrayList<>(),
                    new HashMap<>(),
                    entity.createdAt,
                    entity.updatedAt
                )
            );
    }

    private Uni<Void> persistContextToDatabase(MemoryContext context) {
        return MemorySessionEntity.<MemorySessionEntity>findById(context.getSessionId())
            .onItem().transformToUni(entity -> {
                if (entity == null) {
                    entity = new MemorySessionEntity();
                    entity.sessionId = context.getSessionId();
                    entity.userId = context.getUserId();
                    entity.createdAt = context.getCreatedAt();
                }
                
                entity.metadata = new HashMap<>(context.getMetadata());
                entity.updatedAt = Instant.now();
                
                return entity.persistOrUpdate();
            })
            .replaceWithVoid();
    }

    private Uni<ConversationMemory> createConversationMemory(AgentResponse result) {
        return generateEmbedding(result.getContent())
            .onItem().transform(embedding -> 
                new ConversationMemory(
                    result.getId(),
                    "assistant",
                    result.getContent(),
                    result.getMetadata(),
                    embedding,
                    result.getTimestamp(),
                    null
                )
            );
    }

    private Uni<Void> addMemoryToSession(String sessionId, ConversationMemory memory) {
        ConversationMemoryEntity entity = new ConversationMemoryEntity();
        entity.id = memory.getId();
        entity.sessionId = sessionId;
        entity.role = memory.getRole();
        entity.content = memory.getContent();
        entity.metadata = memory.getMetadata().entrySet().stream()
            .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().toString()));
        entity.embedding = memory.getEmbedding();
        entity.timestamp = memory.getTimestamp();
        entity.relevanceScore = memory.getRelevanceScore();
        
        return entity.persist().replaceWithVoid();
    }

    private Uni<Void> persistExecutionResult(AgentResponse result) {
        ExecutionResultEntity entity = new ExecutionResultEntity();
        entity.id = result.getId();
        entity.sessionId = result.getSessionId();
        entity.content = result.getContent();
        entity.type = result.getType();
        entity.status = result.getStatus();
        entity.metadata = result.getMetadata().entrySet().stream()
            .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().toString()));
        entity.toolCalls = result.getToolCalls();
        entity.timestamp = result.getTimestamp();
        
        return entity.persist().replaceWithVoid();
    }

    private Uni<List<Double>> generateEmbedding(String text) {
        return Uni.createFrom().item(() -> {
            Embedding embedding = embeddingModel.embed(text).content();
            return Arrays.stream(embedding.vector())
                .boxed()
                .collect(Collectors.toList());
        });
    }

    private Uni<ConversationMemory> createSummaryMemory(List<ConversationMemory> memories) {
        String combinedContent = memories.stream()
            .map(ConversationMemory::getContent)
            .collect(Collectors.joining("\n\n"));
        
        // This would typically call an LLM to generate a summary
        String summary = "Summary of " + memories.size() + " previous interactions: " + 
                        combinedContent.substring(0, Math.min(500, combinedContent.length())) + "...";
        
        return generateEmbedding(summary)
            .onItem().transform(embedding -> 
                new ConversationMemory(
                    UUID.randomUUID().toString(),
                    "system",
                    summary,
                    Map.of("type", "summary", "summarized_count", memories.size()),
                    embedding,
                    Instant.now(),
                    1.0
                )
            );
    }

    private double calculateCosineSimilarity(List<Double> vec1, List<Double> vec2) {
        if (vec1.size() != vec2.size()) return 0.0;
        
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (int i = 0; i < vec1.size(); i++) {
            dotProduct += vec1.get(i) * vec2.get(i);
            norm1 += vec1.get(i) * vec1.get(i);
            norm2 += vec2.get(i) * vec2.get(i);
        }
        
        if (norm1 == 0 || norm2 == 0) return 0.0;
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    private ConversationMemory convertToConversationMemory(ConversationMemoryEntity entity, Double relevanceScore) {
        return new ConversationMemory(
            entity.id,
            entity.role,
            entity.content,
            entity.metadata != null ? new HashMap<>(entity.metadata) : new HashMap<>(),
            entity.embedding,
            entity.timestamp,
            relevanceScore != null ? relevanceScore : entity.relevanceScore
        );
    }

    private AgentResponse convertToAgentResponse(ExecutionResultEntity entity) {
        return new AgentResponse(
            entity.id,
            entity.sessionId,
            entity.content,
            entity.type,
            entity.metadata != null ? new HashMap<>(entity.metadata) : new HashMap<>(),
            entity.timestamp,
            entity.status,
            entity.toolCalls
        );
    }

    private Uni<Void> cacheContext(String cacheKey, MemoryContext context) {
        return serializeContext(context)
            .onItem().transformToUni(serialized -> 
                redisAPI.setex(cacheKey, cacheTTL.toSeconds(), serialized)
            )
            .replaceWithVoid();
    }

    private Uni<Void> invalidateCache(String cacheKey) {
        return redisAPI.del(List.of(cacheKey)).replaceWithVoid();
    }

    private Uni<MemoryContext> deserializeContext(String json) {
        // Implementation would deserialize JSON to MemoryContext
        return Uni.createFrom().nullItem(); // Placeholder
    }

    private Uni<String> serializeContext(MemoryContext context) {
        // Implementation would serialize MemoryContext to JSON
        return Uni.createFrom().item("{}"); // Placeholder
    }
}