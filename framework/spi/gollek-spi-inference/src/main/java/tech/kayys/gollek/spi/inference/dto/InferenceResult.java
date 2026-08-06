package tech.kayys.gollek.spi.inference.dto;

import java.util.Map;

/**
 * Modern inference result DTO for high-performance inference.
 */
public record InferenceResult(
    String requestId,
    String model,
    String content,
    int inputTokens,
    int outputTokens,
    long durationMs,
    String finishReason,
    Map<String, Object> extra
) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId;
        private String model;
        private String content;
        private int inputTokens;
        private int outputTokens;
        private long durationMs;
        private String finishReason = "stop";
        private Map<String, Object> extra;

        public Builder requestId(String requestId) { this.requestId = requestId; return this; }
        public Builder model(String model) { this.model = model; return this; }
        public Builder content(String content) { this.content = content; return this; }
        public Builder inputTokens(int inputTokens) { this.inputTokens = inputTokens; return this; }
        public Builder outputTokens(int outputTokens) { this.outputTokens = outputTokens; return this; }
        public Builder durationMs(long durationMs) { this.durationMs = durationMs; return this; }
        public Builder finishReason(String finishReason) { this.finishReason = finishReason; return this; }
        public Builder extra(Map<String, Object> extra) { this.extra = extra; return this; }

        public InferenceResult build() {
            return new InferenceResult(requestId, model, content, inputTokens, outputTokens,
                durationMs, finishReason, extra);
        }
    }
}
