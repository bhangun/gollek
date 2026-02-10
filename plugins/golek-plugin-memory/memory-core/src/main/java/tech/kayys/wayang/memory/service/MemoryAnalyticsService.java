package tech.kayys.wayang.memory.service;

import tech.kayys.wayang.memory.model.*;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@ApplicationScoped
public class MemoryAnalyticsService {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryAnalyticsService.class);
    
    @Inject
    MemoryService memoryService;

    public Uni<MemoryAnalytics> analyzeMemoryUsage(String userId, Duration timeWindow) {
        LOG.info("Analyzing memory usage for user: {} in window: {}", userId, timeWindow);
        
        Instant from = Instant.now().minus(timeWindow);
        
        return getUserSessions(userId, from)
            .onItem().transform(sessions -> {
                int totalSessions = sessions.size();
                long totalMemories = sessions.stream()
                    .mapToLong(session -> session.getConversations().size())
                    .sum();
                
                double avgMemoriesPerSession = totalSessions > 0 ? (double) totalMemories / totalSessions : 0;
                
                Map<String, Long> topicDistribution = extractTopicDistribution(sessions);
                List<UserBehaviorPattern> patterns = identifyBehaviorPatterns(sessions);
                
                return new MemoryAnalytics(
                    userId,
                    timeWindow,
                    totalSessions,
                    totalMemories,
                    avgMemoriesPerSession,
                    topicDistribution,
                    patterns,
                    generateInsights(sessions)
                );
            });
    }

    public Uni<List<MemoryInsight>> generateMemoryInsights(String sessionId) {
        return memoryService.getContext(sessionId, null)
            .onItem().transform(context -> {
                List<MemoryInsight> insights = new java.util.ArrayList<>();
                
                // Analyze conversation patterns
                if (context.getConversations().size() > 10) {
                    insights.add(new MemoryInsight(
                        "HIGH_ACTIVITY",
                        "This session shows high conversation activity",
                        0.8,
                        Map.of("conversationCount", context.getConversations().size())
                    ));
                }
                
                // Analyze topic consistency
                Map<String, Long> topics = extractTopicsFromContext(context);
                if (topics.size() == 1) {
                    insights.add(new MemoryInsight(
                        "FOCUSED_CONVERSATION",
                        "Conversation is focused on a single topic",
                        0.9,
                        Map.of("primaryTopic", topics.keySet().iterator().next())
                    ));
                }
                
                // Analyze temporal patterns
                Duration sessionDuration = Duration.between(
                    context.getCreatedAt(),
                    context.getUpdatedAt()
                );
                
                if (sessionDuration.toHours() > 2) {
                    insights.add(new MemoryInsight(
                        "LONG_SESSION",
                        "Extended conversation session detected",
                        0.7,
                        Map.of("durationHours", sessionDuration.toHours())
                    ));
                }
                
                return insights;
            });
    }

    private Uni<List<MemoryContext>> getUserSessions(String userId, Instant from) {
        // Implementation would query database for user sessions
        return Uni.createFrom().item(List.of());
    }

    private Map<String, Long> extractTopicDistribution(List<MemoryContext> sessions) {
        return sessions.stream()
            .flatMap(session -> session.getConversations().stream())
            .collect(Collectors.groupingBy(
                memory -> extractTopic(memory.getContent()),
                Collectors.counting()
            ));
    }

    private Map<String, Long> extractTopicsFromContext(MemoryContext context) {
        return context.getConversations().stream()
            .collect(Collectors.groupingBy(
                memory -> extractTopic(memory.getContent()),
                Collectors.counting()
            ));
    }

    private String extractTopic(String content) {
        // Simple topic extraction - in production, use NLP libraries
        if (content.toLowerCase().contains("code") || content.toLowerCase().contains("programming")) {
            return "PROGRAMMING";
        } else if (content.toLowerCase().contains("data") || content.toLowerCase().contains("analysis")) {
            return "DATA_ANALYSIS";
        } else if (content.toLowerCase().contains("help") || content.toLowerCase().contains("support")) {
            return "SUPPORT";
        }
        return "GENERAL";
    }

    private List<UserBehaviorPattern> identifyBehaviorPatterns(List<MemoryContext> sessions) {
        List<UserBehaviorPattern> patterns = new java.util.ArrayList<>();
        
        // Pattern 1: Frequent short sessions
        long shortSessions = sessions.stream()
            .filter(session -> session.getConversations().size() < 5)
            .count();
        
        if (shortSessions > sessions.size() * 0.7) {
            patterns.add(new UserBehaviorPattern(
                "FREQUENT_SHORT_SESSIONS",
                "User tends to have many short conversations",
                0.8
            ));
        }
        
        // Pattern 2: Deep dive sessions
        long deepSessions = sessions.stream()
            .filter(session -> session.getConversations().size() > 20)
            .count();
        
        if (deepSessions > 0) {
            patterns.add(new UserBehaviorPattern(
                "DEEP_DIVE_SESSIONS",
                "User occasionally has very detailed conversations",
                0.7
            ));
        }
        
        return patterns;
    }

    private List<MemoryInsight> generateInsights(List<MemoryContext> sessions) {
        List<MemoryInsight> insights = new java.util.ArrayList<>();
        
        if (sessions.size() > 10) {
            insights.add(new MemoryInsight(
                "ACTIVE_USER",
                "User shows high engagement with frequent sessions",
                0.9,
                Map.of("sessionCount", sessions.size())
            ));
        }
        
        return insights;
    }
}