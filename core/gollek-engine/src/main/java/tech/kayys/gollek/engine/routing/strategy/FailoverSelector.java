package tech.kayys.gollek.engine.routing.strategy;

import tech.kayys.gollek.spi.provider.RoutingContext;
import tech.kayys.gollek.spi.routing.RoutingConfig;
import tech.kayys.gollek.spi.provider.LLMProvider;

import java.util.List;

/**
 * Failover selection: uses primary provider with fallback chain.
 * Tries first available provider, returns full fallback list.
 */
public class FailoverSelector implements ProviderSelector {

    @Override
    public ProviderSelection select(
            List<LLMProvider> candidates,
            RoutingContext context,
            RoutingConfig config) {

        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("No candidates available");
        }

        // Prefer user-specified provider if available
        if (context.preferredProvider().isPresent()) {
            String preferred = context.preferredProvider().get();
            for (LLMProvider provider : candidates) {
                if (provider.id().equals(preferred)) {
                    List<LLMProvider> fallbacks = candidates.stream()
                            .filter(p -> !p.id().equals(preferred))
                            .toList();
                    return ProviderSelection.full(provider, 100, null, fallbacks);
                }
            }
        }

        // Use first healthy provider as primary
        LLMProvider primary = candidates.get(0);
        List<LLMProvider> fallbacks = candidates.stream()
                .skip(1)
                .toList();

        return ProviderSelection.full(primary, 100, null, fallbacks);
    }

    @Override
    public String description() {
        return "Primary provider with automatic failover chain";
    }
}
