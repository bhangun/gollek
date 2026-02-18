package tech.kayys.gollek.engine.routing.strategy;

import tech.kayys.gollek.spi.provider.RoutingContext;
import tech.kayys.gollek.spi.routing.RoutingConfig;
import tech.kayys.gollek.spi.provider.LLMProvider;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Random selection with equal probability.
 */
public class RandomSelector implements ProviderSelector {

    @Override
    public ProviderSelection select(
            List<LLMProvider> candidates,
            RoutingContext context,
            RoutingConfig config) {

        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("No candidates available");
        }

        int index = ThreadLocalRandom.current().nextInt(candidates.size());
        LLMProvider selected = candidates.get(index);

        // Build fallback list (random others)
        List<LLMProvider> fallbacks = candidates.stream()
                .filter(p -> !p.id().equals(selected.id()))
                .limit(2)
                .toList();

        return ProviderSelection.withFallbacks(selected, fallbacks);
    }

    @Override
    public String description() {
        return "Random selection with equal probability";
    }
}
