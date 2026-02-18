package tech.kayys.gollek.spi.routing;

import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Configuration for multi-provider routing behavior.
 * Defines pools, strategies, weights, and failover settings.
 */
public record RoutingConfig(
        SelectionStrategy defaultStrategy,
        List<ProviderPool> pools,
        Map<String, Integer> providerWeights,
        boolean autoFailover,
        int maxRetries,
        Duration retryDelay,
        Duration healthCheckInterval,
        boolean preferLocal) {

    public RoutingConfig {
        if (defaultStrategy == null) {
            defaultStrategy = SelectionStrategy.SCORED;
        }
        pools = pools != null ? List.copyOf(pools) : Collections.emptyList();
        providerWeights = providerWeights != null
                ? Map.copyOf(providerWeights)
                : Collections.emptyMap();
        if (maxRetries < 0) {
            maxRetries = 0;
        }
        if (retryDelay == null) {
            retryDelay = Duration.ofMillis(100);
        }
        if (healthCheckInterval == null) {
            healthCheckInterval = Duration.ofSeconds(30);
        }
    }

    /**
     * Get weight for a specific provider (default: 1)
     */
    public int getWeight(String providerId) {
        return providerWeights.getOrDefault(providerId, 1);
    }

    /**
     * Find pool by ID
     */
    public ProviderPool getPool(String poolId) {
        return pools.stream()
                .filter(p -> p.poolId().equals(poolId))
                .findFirst()
                .orElse(null);
    }

    /**
     * Find pools containing a specific provider
     */
    public List<ProviderPool> getPoolsForProvider(String providerId) {
        return pools.stream()
                .filter(p -> p.containsProvider(providerId))
                .toList();
    }

    /**
     * Get all provider IDs across all pools
     */
    public List<String> getAllProviderIds() {
        return pools.stream()
                .flatMap(p -> p.providerIds().stream())
                .distinct()
                .toList();
    }

    /**
     * Get local pools (sorted by priority)
     */
    public List<ProviderPool> getLocalPools() {
        return pools.stream()
                .filter(p -> p.type() == ProviderPool.PoolType.LOCAL)
                .sorted((a, b) -> Integer.compare(b.priority(), a.priority()))
                .toList();
    }

    /**
     * Get cloud pools (sorted by priority)
     */
    public List<ProviderPool> getCloudPools() {
        return pools.stream()
                .filter(p -> p.type() == ProviderPool.PoolType.CLOUD)
                .sorted((a, b) -> Integer.compare(b.priority(), a.priority()))
                .toList();
    }

    /**
     * Create default configuration
     */
    public static RoutingConfig defaults() {
        return new RoutingConfig(
                SelectionStrategy.SCORED,
                Collections.emptyList(),
                Collections.emptyMap(),
                true,
                3,
                Duration.ofMillis(100),
                Duration.ofSeconds(30),
                false);
    }

    /**
     * Builder for custom configuration
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private SelectionStrategy defaultStrategy = SelectionStrategy.SCORED;
        private List<ProviderPool> pools = Collections.emptyList();
        private Map<String, Integer> providerWeights = Collections.emptyMap();
        private boolean autoFailover = true;
        private int maxRetries = 3;
        private Duration retryDelay = Duration.ofMillis(100);
        private Duration healthCheckInterval = Duration.ofSeconds(30);
        private boolean preferLocal = false;

        public Builder defaultStrategy(SelectionStrategy strategy) {
            this.defaultStrategy = strategy;
            return this;
        }

        public Builder pools(List<ProviderPool> pools) {
            this.pools = pools;
            return this;
        }

        public Builder providerWeights(Map<String, Integer> weights) {
            this.providerWeights = weights;
            return this;
        }

        public Builder autoFailover(boolean autoFailover) {
            this.autoFailover = autoFailover;
            return this;
        }

        public Builder maxRetries(int maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }

        public Builder retryDelay(Duration retryDelay) {
            this.retryDelay = retryDelay;
            return this;
        }

        public Builder healthCheckInterval(Duration interval) {
            this.healthCheckInterval = interval;
            return this;
        }

        public Builder preferLocal(boolean preferLocal) {
            this.preferLocal = preferLocal;
            return this;
        }

        public RoutingConfig build() {
            return new RoutingConfig(
                    defaultStrategy, pools, providerWeights, autoFailover,
                    maxRetries, retryDelay, healthCheckInterval, preferLocal);
        }
    }
}
