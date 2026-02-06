package tech.kayys.golek.spi.routing;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Represents a logical group of providers with common characteristics.
 * Pools allow organizing providers by type (cloud/local) or purpose.
 */
public record ProviderPool(
        String poolId,
        String displayName,
        PoolType type,
        List<String> providerIds,
        SelectionStrategy defaultStrategy,
        int priority) {

    /**
     * Pool type classification
     */
    public enum PoolType {
        /**
         * Cloud-based providers (OpenAI, Anthropic, Gemini, etc.)
         */
        CLOUD,

        /**
         * Local inference providers (Ollama, vLLM, custom CPU engine)
         */
        LOCAL,

        /**
         * Mixed pool with both cloud and local providers
         */
        HYBRID
    }

    public ProviderPool {
        Objects.requireNonNull(poolId, "poolId is required");
        Objects.requireNonNull(type, "type is required");
        providerIds = providerIds != null 
            ? List.copyOf(providerIds) 
            : Collections.emptyList();
        if (defaultStrategy == null) {
            defaultStrategy = SelectionStrategy.ROUND_ROBIN;
        }
    }

    /**
     * Check if pool contains a specific provider
     */
    public boolean containsProvider(String providerId) {
        return providerIds.contains(providerId);
    }

    /**
     * Check if pool is empty
     */
    public boolean isEmpty() {
        return providerIds.isEmpty();
    }

    /**
     * Get number of providers in pool
     */
    public int size() {
        return providerIds.size();
    }

    /**
     * Create a cloud provider pool
     */
    public static ProviderPool cloudPool(String poolId, List<String> providers) {
        return new ProviderPool(
            poolId,
            "Cloud Providers",
            PoolType.CLOUD,
            providers,
            SelectionStrategy.WEIGHTED_RANDOM,
            10
        );
    }

    /**
     * Create a local provider pool
     */
    public static ProviderPool localPool(String poolId, List<String> providers) {
        return new ProviderPool(
            poolId,
            "Local Providers",
            PoolType.LOCAL,
            providers,
            SelectionStrategy.LEAST_LOADED,
            20
        );
    }

    /**
     * Builder for custom pools
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String poolId;
        private String displayName;
        private PoolType type = PoolType.HYBRID;
        private List<String> providerIds = Collections.emptyList();
        private SelectionStrategy defaultStrategy = SelectionStrategy.ROUND_ROBIN;
        private int priority = 10;

        public Builder poolId(String poolId) {
            this.poolId = poolId;
            return this;
        }

        public Builder displayName(String displayName) {
            this.displayName = displayName;
            return this;
        }

        public Builder type(PoolType type) {
            this.type = type;
            return this;
        }

        public Builder providerIds(List<String> providerIds) {
            this.providerIds = providerIds;
            return this;
        }

        public Builder defaultStrategy(SelectionStrategy strategy) {
            this.defaultStrategy = strategy;
            return this;
        }

        public Builder priority(int priority) {
            this.priority = priority;
            return this;
        }

        public ProviderPool build() {
            return new ProviderPool(
                poolId, displayName, type, providerIds, defaultStrategy, priority
            );
        }
    }
}
