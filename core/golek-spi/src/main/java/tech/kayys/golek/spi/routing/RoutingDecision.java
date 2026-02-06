package tech.kayys.golek.spi.routing;

import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Result of a routing decision.
 * Contains the selected provider and decision metadata.
 */
public record RoutingDecision(
        String selectedProviderId,
        String poolId,
        SelectionStrategy strategyUsed,
        int score,
        List<String> fallbackProviders,
        Map<String, Object> metadata,
        Instant timestamp) {

    public RoutingDecision {
        fallbackProviders = fallbackProviders != null 
            ? List.copyOf(fallbackProviders) 
            : Collections.emptyList();
        metadata = metadata != null 
            ? Map.copyOf(metadata) 
            : Collections.emptyMap();
        if (timestamp == null) {
            timestamp = Instant.now();
        }
    }

    /**
     * Create a simple decision without fallbacks
     */
    public static RoutingDecision of(String providerId, SelectionStrategy strategy) {
        return new RoutingDecision(
            providerId,
            null,
            strategy,
            100,
            Collections.emptyList(),
            Collections.emptyMap(),
            Instant.now()
        );
    }

    /**
     * Create a decision with fallback providers
     */
    public static RoutingDecision withFallbacks(
            String providerId,
            SelectionStrategy strategy,
            List<String> fallbacks) {
        return new RoutingDecision(
            providerId,
            null,
            strategy,
            100,
            fallbacks,
            Collections.emptyMap(),
            Instant.now()
        );
    }

    /**
     * Check if fallback providers are available
     */
    public boolean hasFallbacks() {
        return !fallbackProviders.isEmpty();
    }

    /**
     * Get next fallback provider
     */
    public String getNextFallback() {
        return fallbackProviders.isEmpty() ? null : fallbackProviders.get(0);
    }

    /**
     * Builder for complex decisions
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String selectedProviderId;
        private String poolId;
        private SelectionStrategy strategyUsed;
        private int score = 100;
        private List<String> fallbackProviders = Collections.emptyList();
        private Map<String, Object> metadata = Collections.emptyMap();
        private Instant timestamp = Instant.now();

        public Builder selectedProviderId(String id) {
            this.selectedProviderId = id;
            return this;
        }

        public Builder poolId(String poolId) {
            this.poolId = poolId;
            return this;
        }

        public Builder strategyUsed(SelectionStrategy strategy) {
            this.strategyUsed = strategy;
            return this;
        }

        public Builder score(int score) {
            this.score = score;
            return this;
        }

        public Builder fallbackProviders(List<String> fallbacks) {
            this.fallbackProviders = fallbacks;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public RoutingDecision build() {
            return new RoutingDecision(
                selectedProviderId, poolId, strategyUsed, score,
                fallbackProviders, metadata, timestamp
            );
        }
    }
}
