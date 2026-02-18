package tech.kayys.gollek.engine.routing.strategy;

import tech.kayys.gollek.spi.provider.RoutingContext;
import tech.kayys.gollek.spi.routing.RoutingConfig;
import tech.kayys.gollek.spi.provider.LLMProvider;

import java.time.Duration;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Latency-optimized selection: prefers fastest providers based on P95 latency.
 */
public class LatencyOptimizedSelector implements ProviderSelector {

    // Cache of P95 latency per provider (in milliseconds)
    private final Map<String, Long> latencyCache = new ConcurrentHashMap<>();
    private static final long DEFAULT_LATENCY_MS = 5000; // 5 seconds default

    @Override
    public ProviderSelection select(
            List<LLMProvider> candidates,
            RoutingContext context,
            RoutingConfig config) {

        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("No candidates available");
        }

        // Sort by latency (ascending)
        List<LLMProvider> sorted = candidates.stream()
                .sorted(Comparator.comparingLong(p -> getLatency(p.id())))
                .toList();

        LLMProvider selected = sorted.get(0);

        // Build fallback list
        List<LLMProvider> fallbacks = sorted.stream()
                .skip(1)
                .limit(2)
                .toList();

        // Score based on latency (lower latency = higher score)
        long latencyMs = getLatency(selected.id());
        int score = calculateScore(latencyMs);

        return ProviderSelection.full(selected, score, null, fallbacks);
    }

    /**
     * Get cached latency for a provider
     */
    public long getLatency(String providerId) {
        return latencyCache.getOrDefault(providerId, DEFAULT_LATENCY_MS);
    }

    /**
     * Update latency for a provider
     */
    public void updateLatency(String providerId, Duration latency) {
        latencyCache.put(providerId, latency.toMillis());
    }

    /**
     * Update latency for a provider (in milliseconds)
     */
    public void updateLatency(String providerId, long latencyMs) {
        latencyCache.put(providerId, latencyMs);
    }

    /**
     * Calculate score from latency (higher = better)
     */
    private int calculateScore(long latencyMs) {
        if (latencyMs < 100) {
            return 100;
        } else if (latencyMs < 500) {
            return 80;
        } else if (latencyMs < 1000) {
            return 60;
        } else if (latencyMs < 3000) {
            return 40;
        } else if (latencyMs < 5000) {
            return 20;
        } else {
            return 10;
        }
    }

    /**
     * Clear latency cache
     */
    public void clearCache() {
        latencyCache.clear();
    }

    @Override
    public String description() {
        return "Selects provider with lowest P95 latency";
    }
}
