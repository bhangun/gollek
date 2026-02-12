package tech.kayys.golek.spi.provider;

import java.time.Duration;
import java.util.Collections;
import java.util.Optional;
import java.util.Set;

import tech.kayys.golek.spi.auth.ApiKeyConstants;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.routing.SelectionStrategy;
import tech.kayys.wayang.tenant.TenantContext;

/**
 * Context for routing decisions.
 * Carries request metadata and routing hints for provider selection.
 */
public record RoutingContext(
        InferenceRequest request,
        TenantContext tenantContext,
        Optional<String> preferredProvider,
        Optional<String> deviceHint,
        Duration timeout,
        boolean costSensitive,
        int priority,
        Optional<SelectionStrategy> strategyOverride,
        Optional<String> poolId,
        Set<String> excludedProviders) {

    public RoutingContext {
        if (timeout == null) {
            timeout = Duration.ofSeconds(30);
        }
        excludedProviders = excludedProviders != null
                ? Set.copyOf(excludedProviders)
                : Collections.emptySet();
    }

    /**
     * Create minimal context for simple routing
     */
    public static RoutingContext simple(
            InferenceRequest request,
            TenantContext tenant) {
        return new RoutingContext(
                request,
                tenant,
                Optional.empty(),
                Optional.empty(),
                Duration.ofSeconds(30),
                false,
                0,
                Optional.empty(),
                Optional.empty(),
                Collections.emptySet());
    }

    /**
     * Create context with preferred provider
     */
    public static RoutingContext withProvider(
            InferenceRequest request,
            TenantContext tenant,
            String providerId) {
        return new RoutingContext(
                request,
                tenant,
                Optional.of(providerId),
                Optional.empty(),
                Duration.ofSeconds(30),
                false,
                0,
                Optional.of(SelectionStrategy.USER_SELECTED),
                Optional.empty(),
                Collections.emptySet());
    }

    /**
     * Create a copy with an excluded provider (for failover)
     */
    public RoutingContext excludeProvider(String providerId) {
        Set<String> newExcluded = new java.util.HashSet<>(excludedProviders);
        newExcluded.add(providerId);
        return new RoutingContext(
                request, tenantContext, preferredProvider, deviceHint,
                timeout, costSensitive, priority, strategyOverride,
                poolId, newExcluded);
    }

    /**
     * Check if a provider is excluded
     */
    public boolean isExcluded(String providerId) {
        return excludedProviders.contains(providerId);
    }

    /**
     * Get effective strategy (override or default)
     */
    public SelectionStrategy getEffectiveStrategy(SelectionStrategy defaultStrategy) {
        return strategyOverride.orElse(defaultStrategy);
    }

    public String apiKey() {
        if (tenantContext == null || tenantContext.getTenantId() == null) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return tenantContext.getTenantId().value();
    }

    /**
     * Builder for complex contexts
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private InferenceRequest request;
        private TenantContext tenantContext;
        private Optional<String> preferredProvider = Optional.empty();
        private Optional<String> deviceHint = Optional.empty();
        private Duration timeout = Duration.ofSeconds(30);
        private boolean costSensitive = false;
        private int priority = 0;
        private Optional<SelectionStrategy> strategyOverride = Optional.empty();
        private Optional<String> poolId = Optional.empty();
        private Set<String> excludedProviders = Collections.emptySet();

        public Builder request(InferenceRequest request) {
            this.request = request;
            return this;
        }

        public Builder tenantContext(TenantContext ctx) {
            this.tenantContext = ctx;
            return this;
        }

        public Builder preferredProvider(String provider) {
            this.preferredProvider = Optional.ofNullable(provider);
            return this;
        }

        public Builder deviceHint(String device) {
            this.deviceHint = Optional.ofNullable(device);
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public Builder costSensitive(boolean costSensitive) {
            this.costSensitive = costSensitive;
            return this;
        }

        public Builder priority(int priority) {
            this.priority = priority;
            return this;
        }

        public Builder strategy(SelectionStrategy strategy) {
            this.strategyOverride = Optional.ofNullable(strategy);
            return this;
        }

        public Builder poolId(String poolId) {
            this.poolId = Optional.ofNullable(poolId);
            return this;
        }

        public Builder excludedProviders(Set<String> excluded) {
            this.excludedProviders = excluded;
            return this;
        }

        public RoutingContext build() {
            return new RoutingContext(
                    request, tenantContext, preferredProvider, deviceHint,
                    timeout, costSensitive, priority, strategyOverride,
                    poolId, excludedProviders);
        }
    }
}
