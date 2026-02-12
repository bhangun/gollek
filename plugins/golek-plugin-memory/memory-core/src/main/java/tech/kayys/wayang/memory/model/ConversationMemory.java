package tech.kayys.wayang.memory.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Represents a single conversation turn in memory
 */
public class ConversationMemory {
    private final String id;
    private final String role;
    private final String content;
    private final Map<String, Object> metadata;
    private final List<Double> embedding;
    private final Instant timestamp;
    private final Double relevanceScore;

    @JsonCreator
    public ConversationMemory(
            @JsonProperty("id") String id,
            @JsonProperty("role") String role,
            @JsonProperty("content") String content,
            @JsonProperty("metadata") Map<String, Object> metadata,
            @JsonProperty("embedding") List<Double> embedding,
            @JsonProperty("timestamp") Instant timestamp,
            @JsonProperty("relevanceScore") Double relevanceScore) {
        this.id = id != null ? id : UUID.randomUUID().toString();
        this.role = role;
        this.content = content;
        this.metadata = metadata;
        this.embedding = embedding;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.relevanceScore = relevanceScore;
    }

    // Getters
    public String getId() { return id; }
    public String getRole() { return role; }
    public String getContent() { return content; }
    public Map<String, Object> getMetadata() { return metadata; }
    public List<Double> getEmbedding() { return embedding; }
    public Instant getTimestamp() { return timestamp; }
    public Double getRelevanceScore() { return relevanceScore; }
}
