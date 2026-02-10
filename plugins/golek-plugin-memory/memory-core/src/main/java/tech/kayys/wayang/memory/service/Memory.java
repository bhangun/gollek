package tech.kayys.gamelan.core.domain;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

/**
 * Represents a memory in the AI system
 */
public class Memory {
    private String id;
    private String namespace;
    private String content;
    private float[] embedding;
    private MemoryType type;
    private Map<String, Object> metadata;
    private Instant timestamp;
    private Instant expiresAt;
    private double importance;

    private Memory(Builder builder) {
        this.id = builder.id;
        this.namespace = builder.namespace;
        this.content = builder.content;
        this.embedding = builder.embedding;
        this.type = builder.type;
        this.metadata = builder.metadata != null ? builder.metadata : new HashMap<>();
        this.timestamp = builder.timestamp != null ? builder.timestamp : Instant.now();
        this.expiresAt = builder.expiresAt;
        this.importance = builder.importance;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getId() {
        return id;
    }

    public String getNamespace() {
        return namespace;
    }

    public String getContent() {
        return content;
    }

    public float[] getEmbedding() {
        return embedding;
    }

    public MemoryType getType() {
        return type;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    public Instant getExpiresAt() {
        return expiresAt;
    }

    public double getImportance() {
        return importance;
    }

    /**
     * Calculate decayed importance based on temporal decay rate
     */
    public double getDecayedImportance(double decayRate) {
        if (timestamp == null)
            return importance;

        long ageMinutes = java.time.Duration.between(timestamp, Instant.now()).toMinutes();
        return importance * Math.exp(-decayRate * ageMinutes);
    }

    public static class Builder {
        private String id;
        private String namespace;
        private String content;
        private float[] embedding;
        private MemoryType type;
        private Map<String, Object> metadata;
        private Instant timestamp;
        private Instant expiresAt;
        private double importance = 0.5;

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder namespace(String namespace) {
            this.namespace = namespace;
            return this;
        }

        public Builder content(String content) {
            this.content = content;
            return this;
        }

        public Builder embedding(float[] embedding) {
            this.embedding = embedding;
            return this;
        }

        public Builder type(MemoryType type) {
            this.type = type;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder addMetadata(String key, Object value) {
            if (this.metadata == null) {
                this.metadata = new HashMap<>();
            }
            this.metadata.put(key, value);
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder expiresAt(Instant expiresAt) {
            this.expiresAt = expiresAt;
            return this;
        }

        public Builder importance(double importance) {
            this.importance = importance;
            return this;
        }

        public Memory build() {
            if (this.id == null) {
                this.id = java.util.UUID.randomUUID().toString();
            }
            return new Memory(this);
        }
    }
}