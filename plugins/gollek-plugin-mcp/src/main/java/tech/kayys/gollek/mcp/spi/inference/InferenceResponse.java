package tech.kayys.gollek.spi.inference;

import java.util.Map;

/**
 * Response from an inference operation.
 */
public class InferenceResponse {

    private final String requestId;
    private final String content;
    private final String model;
    private final int tokensUsed;
    private final long durationMs;
    private final Map<String, Object> metadata;

    private InferenceResponse(Builder builder) {
        this.requestId = builder.requestId;
        this.content = builder.content;
        this.model = builder.model;
        this.tokensUsed = builder.tokensUsed;
        this.durationMs = builder.durationMs;
        this.metadata = Map.copyOf(builder.metadata);
    }

    public String getRequestId() {
        return requestId;
    }

    public String getContent() {
        return content;
    }

    public String getModel() {
        return model;
    }

    public int getTokensUsed() {
        return tokensUsed;
    }

    public long getDurationMs() {
        return durationMs;
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
        private int tokensUsed;
        private long durationMs;
        private final Map<String, Object> metadata = new java.util.HashMap<>();

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

        public Builder durationMs(long durationMs) {
            this.durationMs = durationMs;
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

        public InferenceResponse build() {
            return new InferenceResponse(this);
        }
    }
}