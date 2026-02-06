package tech.kayys.golek.engine.routing.strategy;

import tech.kayys.golek.spi.provider.RoutingContext;
import tech.kayys.golek.spi.routing.RoutingConfig;
import tech.kayys.golek.spi.provider.LLMProvider;

import java.util.List;

/**
 * Strategy interface for provider selection algorithms.
 */
public interface ProviderSelector {

    /**
     * Select a provider from candidates based on the strategy logic.
     *
     * @param candidates Available providers (already filtered for
     *                   health/compatibility)
     * @param context    Routing context with hints and preferences
     * @param config     Global routing configuration
     * @return Selection result with chosen provider and fallbacks
     */
    ProviderSelection select(
            List<LLMProvider> candidates,
            RoutingContext context,
            RoutingConfig config);

    /**
     * Get description of the selection strategy
     */
    default String description() {
        return getClass().getSimpleName();
    }
}
