package tech.kayys.golek.engine.routing.strategy;

import tech.kayys.golek.api.provider.RoutingContext;
import tech.kayys.golek.api.routing.RoutingConfig;
import tech.kayys.golek.api.provider.LLMProvider;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Round-robin selection: cycles through providers sequentially.
 */
public class RoundRobinSelector implements ProviderSelector {

    private final AtomicInteger counter = new AtomicInteger(0);

    @Override
    public ProviderSelection select(
            List<LLMProvider> candidates,
            RoutingContext context,
            RoutingConfig config) {

        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("No candidates available");
        }

        int index = counter.getAndIncrement() % candidates.size();
        LLMProvider selected = candidates.get(index);

        // Build fallback list (remaining providers in order)
        List<LLMProvider> fallbacks = candidates.stream()
                .filter(p -> !p.id().equals(selected.id()))
                .limit(2)
                .toList();

        return ProviderSelection.withFallbacks(selected, fallbacks);
    }

    @Override
    public String description() {
        return "Cycles through providers sequentially";
    }
}
