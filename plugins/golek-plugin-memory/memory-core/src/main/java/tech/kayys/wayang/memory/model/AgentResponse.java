package tech.kayys.wayang.memory.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Response from an agent with memory context
 */
public class AgentResponse {
    private final String id;
    private final String sessionId;
    private final String content;
    private final ResponseType type;
    private final Map<String, Object> metadata;
    private final Instant timestamp;
    private final ResponseStatus status;
    private final List<String> toolCalls;

    @JsonCreator
    public AgentResponse(
            @JsonProperty("id") String id,
            @JsonProperty("sessionId") String sessionId,
            @JsonProperty("content") String content,
            @JsonProperty("type") ResponseType type,
            @JsonProperty("metadata") Map<String, Object> metadata,
            @JsonProperty("timestamp") Instant timestamp,
            @JsonProperty("status") ResponseStatus status,
            @JsonProperty("toolCalls") List<String> toolCalls) {
        this.id = id != null ? id : UUID.randomUUID().toString();
        this.sessionId = sessionId;
        this.content = content;
        this.type = type;
        this.metadata = metadata;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.status = status;
        this.toolCalls = toolCalls;
    }

    // Getters
    public String getId() { return id; }
    public String getSessionId() { return sessionId; }
    public String getContent() { return content; }
    public ResponseType getType() { return type; }
    public Map<String, Object> getMetadata() { return metadata; }
    public Instant getTimestamp() { return timestamp; }
    public ResponseStatus getStatus() { return status; }
    public List<String> getToolCalls() { return toolCalls; }
}
