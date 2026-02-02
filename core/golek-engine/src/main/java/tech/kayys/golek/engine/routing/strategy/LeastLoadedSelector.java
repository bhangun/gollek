package tech.kayys.golek.engine.routing.strategy;

import tech.kayys.golek.api.provider.RoutingContext;
import tech.kayys.golek.api.routing.RoutingConfig;
import tech.kayys.golek.api.provider.LLMProvider;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Least-loaded selection: prefers providers with lowest current load.
 * Tracks active request count per provider.
 */
public class LeastLoadedSelector implements ProviderSelector {

    // Track active requests per provider
    private final Map<String, AtomicInteger> activeRequests = new ConcurrentHashMap<>();

    @Override
    public ProviderSelection select(
            List<LLMProvider> candidates,
            RoutingContext context,
            RoutingConfig config) {

        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("No candidates available");
        }

        // Sort by active request count (ascending)
        List<LLMProvider> sorted = candidates.stream()
                .sorted(Comparator.comparingInt(p -> getActiveCount(p.id())))
                .toList();

        LLMProvider selected = sorted.get(0);

        // Build fallback list (next least loaded)
        List<LLMProvider> fallbacks = sorted.stream()
                .skip(1)
                .limit(2)
                .toList();

        // Calculate score (inverse of load)
        int load = getActiveCount(selected.id());
        int score = Math.max(1, 100 - (load * 10));

        return ProviderSelection.full(selected, score, null, fallbacks);
    }

    /**
     * Get active request count for a provider
     */
    public int getActiveCount(String providerId) {
        return activeRequests.computeIfAbsent(providerId, k -> new AtomicInteger(0)).get();
    }

    /**
     * Increment active count (call when request starts)
     */
    public void incrementActive(String providerId) {
        activeRequests.computeIfAbsent(providerId, k -> new AtomicInteger(0)).incrementAndGet();
    }

    /**
     * Decrement active count (call when request completes)
     */
    public void decrementActive(String providerId) {
        AtomicInteger count = activeRequests.get(providerId);
        if (count != null && count.get() > 0) {
            count.decrementAndGet();
        }
    }

    /**
     * Reset all counters
     */
    public void reset() {
        activeRequests.clear();
    }

    @Override
    public String description() {
        return "Selects provider with lowest active request count";
    }
}
