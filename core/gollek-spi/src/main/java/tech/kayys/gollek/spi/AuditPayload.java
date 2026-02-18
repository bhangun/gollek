package tech.kayys.gollek.spi;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import org.apache.commons.codec.digest.DigestUtils;

import java.time.Instant;
import java.util.*;

/**
 * Audit event for inference operations.
 * Immutable and tamper-evident with hash.
 */
public final class AuditPayload {

    @NotNull
    private final Instant timestamp;

    @NotBlank
    private final String runId;

    private final String nodeId;

    @NotNull
    private final Actor actor;

    @NotBlank
    private final String event;

    @NotBlank
    private final String level;

    private final List<String> tags;
    private final Map<String, Object> metadata;
    private final Map<String, Object> contextSnapshot;

    @NotBlank
    private final String hash;

    @JsonCreator
    public AuditPayload(
            @JsonProperty("timestamp") Instant timestamp,
            @JsonProperty("runId") String runId,
            @JsonProperty("nodeId") String nodeId,
            @JsonProperty("actor") Actor actor,
            @JsonProperty("event") String event,
            @JsonProperty("level") String level,
            @JsonProperty("tags") List<String> tags,
            @JsonProperty("metadata") Map<String, Object> metadata,
            @JsonProperty("contextSnapshot") Map<String, Object> contextSnapshot,
            @JsonProperty("hash") String hash) {
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.runId = Objects.requireNonNull(runId, "runId");
        this.nodeId = nodeId;
        this.actor = Objects.requireNonNull(actor, "actor");
        this.event = Objects.requireNonNull(event, "event");
        this.level = Objects.requireNonNull(level, "level");
        this.tags = tags != null
                ? Collections.unmodifiableList(new ArrayList<>(tags))
                : Collections.emptyList();
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
        this.contextSnapshot = contextSnapshot != null
                ? Collections.unmodifiableMap(new HashMap<>(contextSnapshot))
                : Collections.emptyMap();
        this.hash = Objects.requireNonNull(hash, "hash");
    }

    // Getters
    public Instant getTimestamp() {
        return timestamp;
    }

    public String getRunId() {
        return runId;
    }

    public String getNodeId() {
        return nodeId;
    }

    public Actor getActor() {
        return actor;
    }

    public String getEvent() {
        return event;
    }

    public String getLevel() {
        return level;
    }

    public List<String> getTags() {
        return tags;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public Map<String, Object> getContextSnapshot() {
        return contextSnapshot;
    }

    public String getHash() {
        return hash;
    }

    // Actor record
    public record Actor(
            @JsonProperty("type") String type, // system|human|agent
            @JsonProperty("id") String id,
            @JsonProperty("role") String role) {
        public Actor {
            Objects.requireNonNull(type, "type");
            Objects.requireNonNull(id, "id");
        }

        public static Actor system(String id) {
            return new Actor("system", id, "system");
        }

        public static Actor human(String id, String role) {
            return new Actor("human", id, role);
        }

        public static Actor agent(String id) {
            return new Actor("agent", id, "agent");
        }
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Instant timestamp = Instant.now();
        private String runId;
        private String nodeId;
        private Actor actor = Actor.system("inference-engine");
        private String event;
        private String level = "INFO";
        private final List<String> tags = new ArrayList<>();
        private final Map<String, Object> metadata = new HashMap<>();
        private final Map<String, Object> contextSnapshot = new HashMap<>();

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder runId(String runId) {
            this.runId = runId;
            return this;
        }

        public Builder nodeId(String nodeId) {
            this.nodeId = nodeId;
            return this;
        }

        public Builder actor(Actor actor) {
            this.actor = actor;
            return this;
        }

        public Builder event(String event) {
            this.event = event;
            return this;
        }

        public Builder level(String level) {
            this.level = level;
            return this;
        }

        public Builder tag(String tag) {
            this.tags.add(tag);
            return this;
        }

        public Builder tags(List<String> tags) {
            this.tags.addAll(tags);
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata.putAll(metadata);
            return this;
        }

        public Builder contextSnapshot(Map<String, Object> snapshot) {
            this.contextSnapshot.putAll(snapshot);
            return this;
        }

        public AuditPayload build() {
            Objects.requireNonNull(runId, "runId is required");
            Objects.requireNonNull(event, "event is required");

            String hash = computeHash(
                    timestamp, runId, nodeId, actor.id(), event);

            return new AuditPayload(
                    timestamp, runId, nodeId, actor, event, level,
                    tags, metadata, contextSnapshot, hash);
        }

        private String computeHash(
                Instant timestamp,
                String runId,
                String nodeId,
                String actorId,
                String event) {
            String content = String.join("|",
                    timestamp.toString(),
                    runId,
                    nodeId != null ? nodeId : "",
                    actorId,
                    event);
            return DigestUtils.sha256Hex(content);
        }
    }

    @Override
    public String toString() {
        return "AuditPayload{" + "timestamp=" + timestamp +
                ", event='" + event + '\'' +
                ", level='" + level + '\'' +
                ", runId='" + runId + '\'' +
                '}';
    }
}