package tech.kayys.golek.provider.core.mcp;

import java.time.Duration;
import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

/**
 * Context for tool execution with tracing and metadata.
 */
public final class MCPToolCallContext {

    private final String callId;
    private final TenantContext tenantContext;
    private final String requestId;
    private final Duration timeout;
    private final Map<String, String> headers;
    private final Map<String, Object> metadata;
    private final Instant createdAt;

    private MCPToolCallContext(Builder builder) {
        this.callId = builder.callId;
        this.tenantContext = builder.tenantContext;
        this.requestId = builder.requestId;
        this.timeout = builder.timeout;
        this.headers = Collections.unmodifiableMap(new HashMap<>(builder.headers));
        this.metadata = Collections.unmodifiableMap(new HashMap<>(builder.metadata));
        this.createdAt = builder.createdAt;
    }

    // Getters
    public String getCallId() {
        return callId;
    }

    public TenantContext getTenantContext() {
        return tenantContext;
    }

    public String getRequestId() {
        return requestId;
    }

    public Duration getTimeout() {
        return timeout;
    }

    public Map<String, String> getHeaders() {
        return headers;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String callId = UUID.randomUUID().toString();
        private TenantContext tenantContext;
        private String requestId;
        private Duration timeout = Duration.ofSeconds(30);
        private final Map<String, String> headers = new HashMap<>();
        private final Map<String, Object> metadata = new HashMap<>();
        private Instant createdAt = Instant.now();

        public Builder callId(String callId) {
            this.callId = callId;
            return this;
        }

        public Builder tenantContext(TenantContext tenantContext) {
            this.tenantContext = tenantContext;
            return this;
        }

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public Builder header(String key, String value) {
            this.headers.put(key, value);
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder createdAt(Instant createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public MCPToolCallContext build() {
            Objects.requireNonNull(tenantContext, "tenantContext is required");
            return new MCPToolCallContext(this);
        }
    }

    @Override
    public String toString() {
        return "MCPToolCallContext{callId='" + callId + "', requestId='" + requestId + "'}";
    }
}