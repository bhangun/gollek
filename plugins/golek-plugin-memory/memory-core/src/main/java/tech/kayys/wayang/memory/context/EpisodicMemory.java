package tech.kayys.wayang.memory.context;

import tech.kayys.wayang.memory.model.ConversationMemory;
import java.time.Instant;
import java.util.List;

public class EpisodicMemory {
    private final String sessionId;
    private final List<ConversationMemory> recentMemories;
    private final double coherenceScore;
    private final Instant timestamp;

    public EpisodicMemory(String sessionId, List<ConversationMemory> recentMemories, 
                         double coherenceScore, Instant timestamp) {
        this.sessionId = sessionId;
        this.recentMemories = recentMemories;
        this.coherenceScore = coherenceScore;
        this.timestamp = timestamp;
    }

    public String getSessionId() { return sessionId; }
    public List<ConversationMemory> getRecentMemories() { return recentMemories; }
    public double getCoherenceScore() { return coherenceScore; }
    public Instant getTimestamp() { return timestamp; }
}
