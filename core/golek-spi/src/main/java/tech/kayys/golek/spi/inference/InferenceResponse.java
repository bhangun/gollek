package tech.kayys.golek.spi.inference;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable inference response.
 */
public final class InferenceResponse implements InferenceResponseInterface {

    @NotBlank
    private final String requestId;

    @NotBlank
    private final String content;

    private final String model;
    private final int tokensUsed;
    private final int inputTokens;
    private final int outputTokens;
    private final long durationMs;

    @NotNull
    private final Instant timestamp;

    private final Map<String, Object> metadata;

    private final List<ToolCall> toolCalls;
    private final FinishReason finishReason;
    private final String sessionId;

    /**
     * Reason for stopping generation.
     */
    public enum FinishReason {
        STOP, // Normal completion (EOS token)
        TOOL_CALLS, // Model wants to call tools
        LENGTH, // Hit max_tokens limit
        ERROR // Error during generation
    }

    /**
     * Represents a tool call from the model.
     */
    public record ToolCall(String name, Map<String, Object> arguments) {
        public ToolCall {
            arguments = arguments != null ? Map.copyOf(arguments) : Collections.emptyMap();
        }
    }

    @JsonCreator
    public InferenceResponse(
            @JsonProperty("requestId") String requestId,
            @JsonProperty("content") String content,
            @JsonProperty("model") String model,
            @JsonProperty("tokensUsed") int tokensUsed,
            @JsonProperty("inputTokens") int inputTokens,
            @JsonProperty("outputTokens") int outputTokens,
            @JsonProperty("durationMs") long durationMs,
            @JsonProperty("timestamp") Instant timestamp,
            @JsonProperty("metadata") Map<String, Object> metadata,
            @JsonProperty("toolCalls") List<ToolCall> toolCalls,
            @JsonProperty("finishReason") FinishReason finishReason,
            @JsonProperty("sessionId") String sessionId) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.content = Objects.requireNonNull(content, "content");
        this.model = model;
        this.tokensUsed = tokensUsed > 0 ? tokensUsed : (inputTokens + outputTokens);
        this.inputTokens = inputTokens;
        this.outputTokens = outputTokens;
        this.durationMs = durationMs;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.metadata = metadata != null
                ? Collections.unmodifiableMap(new HashMap<>(metadata))
                : Collections.emptyMap();
        this.toolCalls = toolCalls != null
                ? Collections.unmodifiableList(new ArrayList<>(toolCalls))
                : Collections.emptyList();
        this.finishReason = finishReason != null ? finishReason : FinishReason.STOP;
        this.sessionId = sessionId;
    }

    // Getters
    @Override
    public String getRequestId() {
        return requestId;
    }

    @Override
    public String getContent() {
        return content;
    }

    @Override
    public String getModel() {
        return model;
    }

    @Override
    public int getTokensUsed() {
        return tokensUsed;
    }

    @Override
    public int getInputTokens() {
        return inputTokens;
    }

    @Override
    public int getOutputTokens() {
        return outputTokens;
    }

    @Override
    public long getDurationMs() {
        return durationMs;
    }

    @Override
    public Instant getTimestamp() {
        return timestamp;
    }

    @Override
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public List<ToolCall> getToolCalls() {
        return toolCalls;
    }

    public FinishReason getFinishReason() {
        return finishReason;
    }

    public String getSessionId() {
        return sessionId;
    }

    public boolean hasToolCalls() {
        return toolCalls != null && !toolCalls.isEmpty();
    }

    @Override
    public boolean isStreaming() {
        return false;
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId;
        private String content;
        private String model;
        private int tokensUsed;
        private int inputTokens;
        private int outputTokens;
        private long durationMs;
        private Instant timestamp = Instant.now();
        private final Map<String, Object> metadata = new HashMap<>();
        private final List<ToolCall> toolCalls = new ArrayList<>();
        private FinishReason finishReason = FinishReason.STOP;
        private String sessionId;

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

        public Builder tokensUsed(int tokensUsed) {
            this.tokensUsed = tokensUsed;
            return this;
        }

        public Builder inputTokens(int inputTokens) {
            this.inputTokens = inputTokens;
            return this;
        }

        public Builder outputTokens(int outputTokens) {
            this.outputTokens = outputTokens;
            return this;
        }

        public Builder durationMs(long durationMs) {
            this.durationMs = durationMs;
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

        public Builder toolCall(ToolCall toolCall) {
            this.toolCalls.add(toolCall);
            return this;
        }

        public Builder toolCalls(List<ToolCall> toolCalls) {
            this.toolCalls.addAll(toolCalls);
            return this;
        }

        public Builder finishReason(FinishReason finishReason) {
            this.finishReason = finishReason;
            return this;
        }

        public Builder sessionId(String sessionId) {
            this.sessionId = sessionId;
            return this;
        }

        public InferenceResponse build() {
            Objects.requireNonNull(requestId, "requestId is required");
            Objects.requireNonNull(content, "content is required");
            return new InferenceResponse(
                    requestId, content, model, tokensUsed, inputTokens, outputTokens,
                    durationMs, timestamp, metadata, toolCalls, finishReason, sessionId);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof InferenceResponse that))
            return false;
        return requestId.equals(that.requestId) &&
                content.equals(that.content);
    }

    @Override
    public int hashCode() {
        return Objects.hash(requestId, content);
    }

    @Override
    public String toString() {
        return "InferenceResponse{" +
                "requestId='" + requestId + '\'' +
                ", model='" + model + '\'' +
                ", tokensUsed=" + tokensUsed +
                ", durationMs=" + durationMs +
                '}';
    }
}