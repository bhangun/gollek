package tech.kayys.golek.engine.routing.strategy;

import tech.kayys.golek.api.provider.RoutingContext;
import tech.kayys.golek.api.routing.RoutingConfig;
import tech.kayys.golek.api.provider.LLMProvider;

import java.util.List;

/**
 * User-selected strategy: uses explicit user preference.
 * Fails if preferred provider is not available.
 */
public class UserSelectedSelector implements ProviderSelector {

    @Override
    public ProviderSelection select(
            List<LLMProvider> candidates,
            RoutingContext context,
            RoutingConfig config) {

        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("No candidates available");
        }

        // User must specify a preferred provider
        if (context.preferredProvider().isEmpty()) {
            throw new IllegalArgumentException(
                    "USER_SELECTED strategy requires preferredProvider in context");
        }

        String preferred = context.preferredProvider().get();

        // Find preferred provider in candidates
        for (LLMProvider provider : candidates) {
            if (provider.id().equals(preferred)) {
                // No fallbacks for explicit user selection
                return ProviderSelection.of(provider, 100);
            }
        }

        // Preferred provider not available
        throw new IllegalStateException(
                "Preferred provider '" + preferred + "' is not available. " +
                        "Available providers: " + candidates.stream()
                                .map(LLMProvider::id)
                                .toList());
    }

    @Override
    public String description() {
        return "Uses explicit user-specified provider";
    }
}
