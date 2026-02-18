package tech.kayys.gollek.spi.provider;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import tech.kayys.gollek.spi.tool.ToolCall;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Normalized provider response
 */
public final class ProviderResponse {

    @NotBlank
    private final String requestId;

    @NotBlank
    private final String content;

    private final String model;
    private final String finishReason;
    private final int promptTokens;
    private final int completionTokens;
    private final int totalTokens;
    private final long durationMs;
    private final Instant timestamp;
    private final Map<String, Object> metadata;
    private final int tokensUsed;
    private final List<ToolCall> toolCalls;

    @JsonCreator
    public ProviderResponse(
            @JsonProperty("requestId") String requestId,
            @JsonProperty("content") String content,
            @JsonProperty("model") String model,
            @JsonProperty("finishReason") String finishReason,
            @JsonProperty("promptTokens") int promptTokens,
            @JsonProperty("completionTokens") int completionTokens,
            @JsonProperty("totalTokens") int totalTokens,
            @JsonProperty("tokensUsed") int tokensUsed,
            @JsonProperty("durationMs") long durationMs,

            @JsonProperty("timestamp") Instant timestamp,
            @JsonProperty("toolCalls") List<ToolCall> toolCalls,
            @JsonProperty("metadata") Map<String, Object> metadata) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.content = Objects.requireNonNull(content, "content");
        this.model = model;
        this.finishReason = finishReason;
        this.promptTokens = promptTokens;
        this.completionTokens = completionTokens;
        this.totalTokens = totalTokens;
        this.tokensUsed = tokensUsed;
        this.durationMs = durationMs;
        this.toolCalls = toolCalls != null
                ? Collections.unmodifiableList(new ArrayList<>(toolCalls))
                : Collections.emptyList();
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
    }

    // Getters
    public String getRequestId() {
        return requestId;
    }

    public String getContent() {
        return content;
    }

    public String getModel() {
        return model;
    }

    public String getFinishReason() {
        return finishReason;
    }

    public int getPromptTokens() {
        return promptTokens;
    }

    public int getCompletionTokens() {
        return completionTokens;
    }

    public int getTotalTokens() {
        return totalTokens;
    }

    public long getDurationMs() {
        return durationMs;
    }

    public int getTokensUsed() {
        return tokensUsed;
    }

    public List<ToolCall> getToolCalls() {
        return toolCalls;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId;
        private String content;
        private String model;
        private String finishReason = "stop";
        private int promptTokens;
        private int completionTokens;
        private int totalTokens;
        private long durationMs;
        private int tokensUsed;
        private List<ToolCall> toolCalls = new ArrayList<>();
        private Instant timestamp = Instant.now();
        private final Map<String, Object> metadata = new HashMap<>();

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder content(String content) {
            this.content = content;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder finishReason(String finishReason) {
            this.finishReason = finishReason;
            return this;
        }

        public Builder promptTokens(int promptTokens) {
            this.promptTokens = promptTokens;
            return this;
        }

        public Builder completionTokens(int completionTokens) {
            this.completionTokens = completionTokens;
            return this;
        }

        public Builder totalTokens(int totalTokens) {
            this.totalTokens = totalTokens;
            return this;
        }

        public Builder durationMs(long durationMs) {
            this.durationMs = durationMs;
            return this;
        }

        public Builder tokensUsed(int tokensUsed) {
            this.tokensUsed = tokensUsed;
            return this;
        }

        public Builder toolCall(ToolCall toolCall) {
            this.toolCalls.add(toolCall);
            return this;
        }

        public Builder toolCalls(List<ToolCall> toolCalls) {
            if (toolCalls != null) {
                this.toolCalls.addAll(toolCalls);
            }
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
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

        public ProviderResponse build() {
            Objects.requireNonNull(requestId, "requestId is required");
            Objects.requireNonNull(content, "content is required");
            return new ProviderResponse(
                    requestId, content, model, finishReason,
                    promptTokens, completionTokens, totalTokens, tokensUsed,
                    durationMs, timestamp, toolCalls, metadata);
        }
    }

    @Override
    public String toString() {
        return "ProviderResponse{" +
                "requestId='" + requestId + '\'' +
                ", model='" + model + '\'' +
                ", tokens=" + totalTokens +
                ", durationMs=" + durationMs +
                '}';
    }
}