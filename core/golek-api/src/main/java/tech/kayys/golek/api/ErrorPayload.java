package tech.kayys.golek.api;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Standardized error payload for all inference failures.
 * Integrates with golek's error-as-input pattern.
 */
public final class ErrorPayload {

    @NotBlank
    private final String type;

    @NotBlank
    private final String message;

    private final Map<String, Object> details;
    private final boolean retryable;
    private final String originNode;
    private final String originRunId;
    private final int attempt;
    private final int maxAttempts;

    @NotNull
    private final Instant timestamp;

    private final String suggestedAction;
    private final String provenanceRef;

    @JsonCreator
    public ErrorPayload(
            @JsonProperty("type") String type,
            @JsonProperty("message") String message,
            @JsonProperty("details") Map<String, Object> details,
            @JsonProperty("retryable") boolean retryable,
            @JsonProperty("originNode") String originNode,
            @JsonProperty("originRunId") String originRunId,
            @JsonProperty("attempt") int attempt,
            @JsonProperty("maxAttempts") int maxAttempts,
            @JsonProperty("timestamp") Instant timestamp,
            @JsonProperty("suggestedAction") String suggestedAction,
            @JsonProperty("provenanceRef") String provenanceRef) {
        this.type = Objects.requireNonNull(type, "type");
        this.message = Objects.requireNonNull(message, "message");
        this.details = details != null
                ? Collections.unmodifiableMap(new HashMap<>(details))
                : Collections.emptyMap();
        this.retryable = retryable;
        this.originNode = originNode;
        this.originRunId = originRunId;
        this.attempt = attempt;
        this.maxAttempts = maxAttempts;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.suggestedAction = suggestedAction;
        this.provenanceRef = provenanceRef;
    }

    // Getters
    public String getType() {
        return type;
    }

    public String getMessage() {
        return message;
    }

    public Map<String, Object> getDetails() {
        return details;
    }

    public boolean isRetryable() {
        return retryable;
    }

    public String getOriginNode() {
        return originNode;
    }

    public String getOriginRunId() {
        return originRunId;
    }

    public int getAttempt() {
        return attempt;
    }

    public int getMaxAttempts() {
        return maxAttempts;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    public String getSuggestedAction() {
        return suggestedAction;
    }

    public String getProvenanceRef() {
        return provenanceRef;
    }

    // Factory methods
    public static ErrorPayload from(Throwable error, String nodeId, String runId) {
        return builder()
                .type(error.getClass().getSimpleName())
                .message(error.getMessage() != null ? error.getMessage() : "Unknown error")
                .originNode(nodeId)
                .originRunId(runId)
                .retryable(isRetryable(error))
                .suggestedAction(determineSuggestedAction(error))
                .details(extractDetails(error))
                .build();
    }

    private static boolean isRetryable(Throwable error) {
        String className = error.getClass().getName();
        return !className.contains("Validation") &&
                !className.contains("Authorization") &&
                !className.contains("Quota");
    }

    private static String determineSuggestedAction(Throwable error) {
        String className = error.getClass().getName();
        if (className.contains("Quota")) {
            return "escalate";
        } else if (className.contains("Provider")) {
            return "retry";
        } else if (className.contains("Validation")) {
            return "human_review";
        } else {
            return "fallback";
        }
    }

    private static Map<String, Object> extractDetails(Throwable error) {
        Map<String, Object> details = new HashMap<>();
        details.put("errorClass", error.getClass().getName());
        if (error.getCause() != null) {
            details.put("cause", error.getCause().getMessage());
        }
        return details;
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String type;
        private String message;
        private final Map<String, Object> details = new HashMap<>();
        private boolean retryable = false;
        private String originNode;
        private String originRunId;
        private int attempt = 0;
        private int maxAttempts = 3;
        private Instant timestamp = Instant.now();
        private String suggestedAction;
        private String provenanceRef;

        public Builder type(String type) {
            this.type = type;
            return this;
        }

        public Builder message(String message) {
            this.message = message;
            return this;
        }

        public Builder detail(String key, Object value) {
            if (value != null) {
                this.details.put(key, value);
            }
            return this;
        }

        public Builder details(Map<String, Object> details) {
            this.details.putAll(details);
            return this;
        }

        public Builder retryable(boolean retryable) {
            this.retryable = retryable;
            return this;
        }

        public Builder originNode(String originNode) {
            this.originNode = originNode;
            return this;
        }

        public Builder originRunId(String originRunId) {
            this.originRunId = originRunId;
            return this;
        }

        public Builder attempt(int attempt) {
            this.attempt = attempt;
            return this;
        }

        public Builder maxAttempts(int maxAttempts) {
            this.maxAttempts = maxAttempts;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder suggestedAction(String suggestedAction) {
            this.suggestedAction = suggestedAction;
            return this;
        }

        public Builder provenanceRef(String provenanceRef) {
            this.provenanceRef = provenanceRef;
            return this;
        }

        public ErrorPayload build() {
            Objects.requireNonNull(type, "type is required");
            Objects.requireNonNull(message, "message is required");
            return new ErrorPayload(
                    type, message, details, retryable, originNode, originRunId,
                    attempt, maxAttempts, timestamp, suggestedAction, provenanceRef);
        }
    }

    @Override
    public String toString() {
        return "ErrorPayload{" +
                "type='" + type + '\'' +
                ", message='" + message + '\'' +
                ", retryable=" + retryable +
                ", attempt=" + attempt +
                ", suggestedAction='" + suggestedAction + '\'' +
                '}';
    }
}