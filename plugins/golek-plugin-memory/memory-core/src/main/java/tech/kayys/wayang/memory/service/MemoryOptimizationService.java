package tech.kayys.wayang.memory.service;

import tech.kayys.wayang.memory.model.ConversationMemory;
import tech.kayys.wayang.memory.model.MemoryContext;
import tech.kayys.wayang.memory.model.OptimizationResult;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.stream.Collectors;

@ApplicationScoped
public class MemoryOptimizationService {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryOptimizationService.class);
    
    @Inject
    MemoryService memoryService;
    
    @ConfigProperty(name = "memory.optimization.relevance.threshold", defaultValue = "0.5")
    double relevanceThreshold;
    
    @ConfigProperty(name = "memory.optimization.age.threshold", defaultValue = "P7D")
    Duration ageThreshold;

    public Uni<OptimizationResult> optimizeMemory(String sessionId) {
        LOG.info("Optimizing memory for session: {}", sessionId);
        
        return memoryService.getContext(sessionId, null)
            .onItem().transformToUni(context -> {
                List<ConversationMemory> optimizedMemories = optimizeConversations(context.getConversations());
                
                MemoryContext optimizedContext = new MemoryContext(
                    context.getSessionId(),
                    context.getUserId(),
                    optimizedMemories,
                    context.getMetadata(),
                    context.getCreatedAt(),
                    Instant.now()
                );
                
                return memoryService.storeContext(optimizedContext)
                    .onItem().transform(unused -> new OptimizationResult(
                        sessionId,
                        context.getConversations().size(),
                        optimizedMemories.size(),
                        calculateSpaceSaved(context.getConversations(), optimizedMemories),
                        Instant.now()
                    ));
            });
    }

    private List<ConversationMemory> optimizeConversations(List<ConversationMemory> conversations) {
        Instant cutoffTime = Instant.now().minus(ageThreshold);
        
        return conversations.stream()
            .filter(memory -> {
                // Keep recent memories
                if (memory.getTimestamp().isAfter(cutoffTime)) {
                    return true;
                }
                
                // Keep high-relevance memories
                if (memory.getRelevanceScore() != null && memory.getRelevanceScore() > relevanceThreshold) {
                    return true;
                }
                
                // Keep summary memories
                if (memory.getMetadata().containsKey("type") && "summary".equals(memory.getMetadata().get("type"))) {
                    return true;
                }
                
                return false;
            })
            .collect(Collectors.toList());
    }

    private long calculateSpaceSaved(List<ConversationMemory> original, List<ConversationMemory> optimized) {
        long originalSize = original.stream()
            .mapToLong(memory -> memory.getContent().length())
            .sum();
        
        long optimizedSize = optimized.stream()
            .mapToLong(memory -> memory.getContent().length())
            .sum();
        
        return originalSize - optimizedSize;
    }
}