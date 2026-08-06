package tech.kayys.gollek.spi.inference.dto;

/**
 * Modern inference context record for high-performance inference.
 */
public record InferenceContext(
    String apiKey,
    String requestId,
    String tenantId,
    String traceId
) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String apiKey;
        private String requestId = java.util.UUID.randomUUID().toString();
        private String tenantId = "default";
        private String traceId;

        public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }
        public Builder requestId(String requestId) { this.requestId = requestId; return this; }
        public Builder tenantId(String tenantId) { this.tenantId = tenantId; return this; }
        public Builder traceId(String traceId) { this.traceId = traceId; return this; }

        public InferenceContext build() {
            return new InferenceContext(apiKey, requestId, tenantId, traceId != null ? traceId : requestId);
        }
    }
}
