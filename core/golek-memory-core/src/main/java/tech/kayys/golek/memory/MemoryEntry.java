package tech.kayys.golek.memory;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import java.time.Instant;
import java.util.*;

/**
 * Immutable memory entry.
 * Represents a single piece of stored memory across all memory types.
 */
public final class MemoryEntry {

    @NotBlank
    private final String id;

    @NotNull
    private final MemoryType type;

    @NotBlank
    private final String content;

    private final String summary; // Optional condensed version
    private final List<Double> embedding; // Vector representation
    private final Map<String, Object> metadata;
    private final Set<String> tags;

    @NotNull
    private final Instant createdAt;

    private final Instant expiresAt; // Optional TTL
    private final String tenantId;
    private final String userId;
    private final String sessionId;

    private final double importance; // 0.0 - 1.0
    private final int accessCount;
    private final Instant lastAccessedAt;

    @JsonCreator
    public MemoryEntry(
            @JsonProperty("id") String id,
            @JsonProperty("type") MemoryType type,
            @JsonProperty("content") String content,
            @JsonProperty("summary") String summary,
            @JsonProperty("embedding") List<Double> embedding,
            @JsonProperty("metadata") Map<String, Object> metadata,
            @JsonProperty("tags") Set<String> tags,
            @JsonProperty("createdAt") Instant createdAt,
            @JsonProperty("expiresAt") Instant expiresAt,
            @JsonProperty("tenantId") String tenantId,
            @JsonProperty("userId") String userId,
            @JsonProperty("sessionId") String sessionId,
            @JsonProperty("importance") double importance,
            @JsonProperty("accessCount") int accessCount,
            @JsonProperty("lastAccessedAt") Instant lastAccessedAt) {
        this.id = Objects.requireNonNull(id, "id");
        this.type = Objects.requireNonNull(type, "type");
        this.content = Objects.requireNonNull(content, "content");
        this.summary = summary;
        this.embedding = embedding != null
                ? Collections.unmodifiableList(new ArrayList<>(embedding))
                : null;
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
        this.tags = tags != null
                ? Collections.unmodifiableSet(new HashSet<>(tags))
                : Collections.emptySet();
        this.createdAt = createdAt != null ? createdAt : Instant.now();
        this.expiresAt = expiresAt;
        this.tenantId = tenantId;
        this.userId = userId;
        this.sessionId = sessionId;
        this.importance = Math.max(0.0, Math.min(1.0, importance));
        this.accessCount = Math.max(0, accessCount);
        this.lastAccessedAt = lastAccessedAt;
    }

    // Getters
    public String getId() {
        return id;
    }

    public MemoryType getType() {
        return type;
    }

    public String getContent() {
        return content;
    }

    public String getSummary() {
        return summary;
    }

    public List<Double> getEmbedding() {
        return embedding;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public Set<String> getTags() {
        return tags;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Optional<Instant> getExpiresAt() {
        return Optional.ofNullable(expiresAt);
    }

    public String getTenantId() {
        return tenantId;
    }

    public String getUserId() {
        return userId;
    }

    public String getSessionId() {
        return sessionId;
    }

    public double getImportance() {
        return importance;
    }

    public int getAccessCount() {
        return accessCount;
    }

    public Instant getLastAccessedAt() {
        return lastAccessedAt;
    }

    /**
     * Check if memory has expired
     */
    public boolean isExpired() {
        return expiresAt != null && Instant.now().isAfter(expiresAt);
    }

    /**
     * Check if memory has embedding
     */
    public boolean hasEmbedding() {
        return embedding != null && !embedding.isEmpty();
    }

    /**
     * Get display content (summary if available, else content)
     */
    public String getDisplayContent() {
        return summary != null ? summary : content;
    }

    /**
     * Calculate recency score (1.0 = just created, 0.0 = very old)
     */
    public double getRecencyScore() {
        long ageSeconds = Instant.now().getEpochSecond() - createdAt.getEpochSecond();
        long maxAge = 30 * 24 * 60 * 60; // 30 days
        return Math.max(0.0, 1.0 - ((double) ageSeconds / maxAge));
    }

    /**
     * Calculate frequency score based on access count
     */
    public double getFrequencyScore() {
        return Math.min(1.0, accessCount / 100.0);
    }

    /**
     * Create updated entry with incremented access
     */
    public MemoryEntry withAccess() {
        return toBuilder()
                .accessCount(accessCount + 1)
                .lastAccessedAt(Instant.now())
                .build();
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        return new Builder()
                .id(id)
                .type(type)
                .content(content)
                .summary(summary)
                .embedding(embedding)
                .metadata(metadata)
                .tags(tags)
                .createdAt(createdAt)
                .expiresAt(expiresAt)
                .tenantId(tenantId)
                .userId(userId)
                .sessionId(sessionId)
                .importance(importance)
                .accessCount(accessCount)
                .lastAccessedAt(lastAccessedAt);
    }

    public static class Builder {
        private String id = UUID.randomUUID().toString();
        private MemoryType type;
        private String content;
        private String summary;
        private List<Double> embedding;
        private final Map<String, Object> metadata = new HashMap<>();
        private final Set<String> tags = new HashSet<>();
        private Instant createdAt = Instant.now();
        private Instant expiresAt;
        private String tenantId;
        private String userId;
        private String sessionId;
        private double importance = 0.5;
        private int accessCount = 0;
        private Instant lastAccessedAt = Instant.now();

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder type(MemoryType type) {
            this.type = type;
            return this;
        }

        public Builder content(String content) {
            this.content = content;
            return this;
        }

        public Builder summary(String summary) {
            this.summary = summary;
            return this;
        }

        public Builder embedding(List<Double> embedding) {
            this.embedding = embedding;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata.putAll(metadata);
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder tags(Set<String> tags) {
            this.tags.addAll(tags);
            return this;
        }

        public Builder tag(String tag) {
            this.tags.add(tag);
            return this;
        }

        public Builder createdAt(Instant createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public Builder expiresAt(Instant expiresAt) {
            this.expiresAt = expiresAt;
            return this;
        }

        public Builder tenantId(String tenantId) {
            this.tenantId = tenantId;
            return this;
        }

        public Builder userId(String userId) {
            this.userId = userId;
            return this;
        }

        public Builder sessionId(String sessionId) {
            this.sessionId = sessionId;
            return this;
        }

        public Builder importance(double importance) {
            this.importance = importance;
            return this;
        }

        public Builder accessCount(int accessCount) {
            this.accessCount = accessCount;
            return this;
        }

        public Builder lastAccessedAt(Instant lastAccessedAt) {
            this.lastAccessedAt = lastAccessedAt;
            return this;
        }

        public MemoryEntry build() {
            Objects.requireNonNull(type, "type is required");
            Objects.requireNonNull(content, "content is required");
            return new MemoryEntry(
                    id, type, content, summary, embedding, metadata, tags,
                    createdAt, expiresAt, tenantId, userId, sessionId,
                    importance, accessCount, lastAccessedAt);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof MemoryEntry that))
            return false;
        return id.equals(that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "MemoryEntry{" +
                "id='" + id + '\'' +
                ", type=" + type +
                ", importance=" + importance +
                ", createdAt=" + createdAt +
                '}';
    }
}