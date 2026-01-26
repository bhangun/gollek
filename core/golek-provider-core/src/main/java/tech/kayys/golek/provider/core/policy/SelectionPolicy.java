package tech.kayys.golek.provider.core.policy;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.golek.api.model.ModelManifest;
import tech.kayys.golek.api.provider.RoutingContext;
import tech.kayys.golek.provider.core.spi.LLMProvider;
import tech.kayys.golek.provider.core.spi.ProviderCandidate;

import java.util.List;

/**
 * Selection policy for provider ranking
 */
@ApplicationScoped
public class SelectionPolicy {

    /**
     * Rank providers based on criteria
     */
    public List<ProviderCandidate> rankProviders(
            ModelManifest manifest,
            RoutingContext context,
            List<LLMProvider> providers) {
        // Implemented in ModelRouterService
        return List.of();
    }
}
