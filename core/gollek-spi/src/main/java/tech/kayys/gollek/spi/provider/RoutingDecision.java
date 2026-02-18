package tech.kayys.gollek.spi.provider;

import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.routing.SelectionStrategy;

import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Result of a provider routing decision.
 */
public record RoutingDecision(
        String providerId,
        LLMProvider provider,
        int score,
        List<String> fallbackProviders,
        ModelManifest manifest,
        RoutingContext context,
        String poolId,
        SelectionStrategy strategyUsed,
        Map<String, Object> metadata,
        Instant timestamp) {

    public RoutingDecision {
        fallbackProviders = fallbackProviders != null ? List.copyOf(fallbackProviders) : Collections.emptyList();
        metadata = metadata != null ? Map.copyOf(metadata) : Collections.emptyMap();
        if (timestamp == null) {
            timestamp = Instant.now();
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String providerId;
        private LLMProvider provider;
        private int score;
        private List<String> fallbackProviders = Collections.emptyList();
        private ModelManifest manifest;
        private RoutingContext context;
        private String poolId;
        private SelectionStrategy strategyUsed;
        private Map<String, Object> metadata = Collections.emptyMap();
        private Instant timestamp = Instant.now();

        public Builder providerId(String providerId) {
            this.providerId = providerId;
            return this;
        }

        public Builder provider(LLMProvider provider) {
            this.provider = provider;
            return this;
        }

        public Builder score(int score) {
            this.score = score;
            return this;
        }

        public Builder fallbackProviders(List<String> fallbackProviders) {
            this.fallbackProviders = fallbackProviders;
            return this;
        }

        public Builder manifest(ModelManifest manifest) {
            this.manifest = manifest;
            return this;
        }

        public Builder context(RoutingContext context) {
            this.context = context;
            return this;
        }

        public Builder poolId(String poolId) {
            this.poolId = poolId;
            return this;
        }

        public Builder strategyUsed(SelectionStrategy strategyUsed) {
            this.strategyUsed = strategyUsed;
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

        public Builder selectedProviderId(String id) {
            this.providerId = id;
            return this;
        }

        public RoutingDecision build() {
            return new RoutingDecision(
                    providerId, provider, score, fallbackProviders,
                    manifest, context, poolId, strategyUsed, metadata, timestamp);
        }
    }
}
