package tech.kayys.golek.engine.provider.adapter;

import java.time.Instant;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import tech.kayys.wayang.tenant.TenantContext;

/**
 * Context object passed to providers during execution
 */
public final class ProviderContext {

    private final String requestId;
    private final TenantContext tenantContext;
    private final Instant startTime;
    private final Map<String, Object> attributes;

    public ProviderContext(String requestId, TenantContext tenantContext) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.tenantContext = tenantContext;
        this.startTime = Instant.now();
        this.attributes = new ConcurrentHashMap<>();
    }

    public String getRequestId() {
        return requestId;
    }

    public TenantContext getTenantContext() {
        return tenantContext;
    }

    public Instant getStartTime() {
        return startTime;
    }

    public long getElapsedMs() {
        return Instant.now().toEpochMilli() - startTime.toEpochMilli();
    }

    public void setAttribute(String key, Object value) {
        attributes.put(key, value);
    }

    @SuppressWarnings("unchecked")
    public <T> Optional<T> getAttribute(String key, Class<T> type) {
        Object value = attributes.get(key);
        if (value != null && type.isInstance(value)) {
            return Optional.of((T) value);
        }
        return Optional.empty();
    }

    public Map<String, Object> getAttributes() {
        return Map.copyOf(attributes);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId;
        private TenantContext tenantContext;

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder tenantContext(TenantContext tenantContext) {
            this.tenantContext = tenantContext;
            return this;
        }

        public ProviderContext build() {
            return new ProviderContext(requestId, tenantContext);
        }
    }

    @Override
    public String toString() {
        return "ProviderContext{" +
                "requestId='" + requestId + '\'' +
                ", tenant=" + (tenantContext != null ? tenantContext.getTenantId() : "null") +
                ", elapsed=" + getElapsedMs() + "ms" +
                '}';
    }
}