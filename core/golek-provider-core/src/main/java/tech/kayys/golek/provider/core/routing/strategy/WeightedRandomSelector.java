package tech.kayys.golek.provider.core.routing.strategy;

import tech.kayys.golek.api.provider.RoutingContext;
import tech.kayys.golek.api.routing.RoutingConfig;
import tech.kayys.golek.provider.core.spi.LLMProvider;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Weighted random selection based on configured weights.
 * Higher weight = higher selection probability.
 */
public class WeightedRandomSelector implements ProviderSelector {

    @Override
    public ProviderSelection select(
            List<LLMProvider> candidates,
            RoutingContext context,
            RoutingConfig config) {

        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("No candidates available");
        }

        // Calculate total weight
        int totalWeight = candidates.stream()
                .mapToInt(p -> config.getWeight(p.id()))
                .sum();

        if (totalWeight == 0) {
            totalWeight = candidates.size(); // Default to equal weights
        }

        // Random selection based on weight
        int random = ThreadLocalRandom.current().nextInt(totalWeight);
        int cumulative = 0;

        LLMProvider selected = candidates.get(0);
        for (LLMProvider provider : candidates) {
            cumulative += config.getWeight(provider.id());
            if (random < cumulative) {
                selected = provider;
                break;
            }
        }

        // Build fallback list (by weight order)
        final LLMProvider finalSelected = selected;
        List<LLMProvider> fallbacks = candidates.stream()
                .filter(p -> !p.id().equals(finalSelected.id()))
                .sorted((a, b) -> Integer.compare(
                        config.getWeight(b.id()),
                        config.getWeight(a.id())))
                .limit(2)
                .toList();

        return ProviderSelection.full(
                selected,
                config.getWeight(selected.id()) * 10,
                null,
                fallbacks);
    }

    @Override
    public String description() {
        return "Random selection weighted by configuration";
    }
}
