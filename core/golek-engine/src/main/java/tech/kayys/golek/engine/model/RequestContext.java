package tech.kayys.golek.engine.model;

import tech.kayys.golek.api.tenant.TenantContext;

import java.time.Duration;

public record RequestContext(
        TenantContext tenantContext,
        Duration timeout,
        int priority,
        String preferredDevice
) {
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private TenantContext tenantContext;
        private Duration timeout;
        private int priority = 0;
        private String preferredDevice;

        public Builder tenantContext(TenantContext tenantContext) {
            this.tenantContext = tenantContext;
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public Builder priority(int priority) {
            this.priority = priority;
            return this;
        }

        public Builder preferredDevice(String preferredDevice) {
            this.preferredDevice = preferredDevice;
            return this;
        }

        public RequestContext build() {
            return new RequestContext(tenantContext, timeout, priority, preferredDevice);
        }
    }
}