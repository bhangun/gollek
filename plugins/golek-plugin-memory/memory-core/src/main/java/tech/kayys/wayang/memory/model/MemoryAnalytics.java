package tech.kayys.wayang.memory.model;

import java.time.Duration;
import java.util.List;
import java.util.Map;

public class MemoryAnalytics {
    private final String userId;
    private final Duration timeWindow;
    private final int totalSessions;
    private final long totalMemories;
    private final double avgMemoriesPerSession;
    private final Map<String, Long> topicDistribution;
    private final List<UserBehaviorPattern> behaviorPatterns;
    private final List<MemoryInsight> insights;

    public MemoryAnalytics(
            String userId,
            Duration timeWindow,
            int totalSessions,
            long totalMemories,
            double avgMemoriesPerSession,
            Map<String, Long> topicDistribution,
            List<UserBehaviorPattern> behaviorPatterns,
            List<MemoryInsight> insights) {
        this.userId = userId;
        this.timeWindow = timeWindow;
        this.totalSessions = totalSessions;
        this.totalMemories = totalMemories;
        this.avgMemoriesPerSession = avgMemoriesPerSession;
        this.topicDistribution = topicDistribution;
        this.behaviorPatterns = behaviorPatterns;
        this.insights = insights;
    }

    // Getters
    public String getUserId() { return userId; }
    public Duration getTimeWindow() { return timeWindow; }
    public int getTotalSessions() { return totalSessions; }
    public long getTotalMemories() { return totalMemories; }
    public double getAvgMemoriesPerSession() { return avgMemoriesPerSession; }
    public Map<String, Long> getTopicDistribution() { return topicDistribution; }
    public List<UserBehaviorPattern> getBehaviorPatterns() { return behaviorPatterns; }
    public List<MemoryInsight> getInsights() { return insights; }
}