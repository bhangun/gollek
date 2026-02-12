package tech.kayys.wayang.memory.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * Context for a memory session
 */
public class MemoryContext {
    private final String sessionId;
    private final String userId;
    private final List<ConversationMemory> conversations;
    private final Map<String, Object> metadata;
    private final Instant createdAt;
    private final Instant updatedAt;

    @JsonCreator
    public MemoryContext(
            @JsonProperty("sessionId") String sessionId,
            @JsonProperty("userId") String userId,
            @JsonProperty("conversations") List<ConversationMemory> conversations,
            @JsonProperty("metadata") Map<String, Object> metadata,
            @JsonProperty("createdAt") Instant createdAt,
            @JsonProperty("updatedAt") Instant updatedAt) {
        this.sessionId = sessionId;
        this.userId = userId;
        this.conversations = conversations;
        this.metadata = metadata;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }

    // Getters
    public String getSessionId() { return sessionId; }
    public String getUserId() { return userId; }
    public List<ConversationMemory> getConversations() { return conversations; }
    public Map<String, Object> getMetadata() { return metadata; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
}
