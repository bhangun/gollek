package tech.kayys.gollek.engine.routing.strategy;

import tech.kayys.gollek.spi.provider.LLMProvider;

import java.util.Collections;
import java.util.List;

/**
 * Result of a provider selection operation.
 */
public record ProviderSelection(
        LLMProvider provider,
        int score,
        String poolId,
        List<LLMProvider> fallbacks) {

    public ProviderSelection {
        fallbacks = fallbacks != null ? List.copyOf(fallbacks) : Collections.emptyList();
    }

    /**
     * Create a simple selection without fallbacks
     */
    public static ProviderSelection of(LLMProvider provider) {
        return new ProviderSelection(provider, 100, null, Collections.emptyList());
    }

    /**
     * Create a selection with score
     */
    public static ProviderSelection of(LLMProvider provider, int score) {
        return new ProviderSelection(provider, score, null, Collections.emptyList());
    }

    /**
     * Create a selection with fallbacks
     */
    public static ProviderSelection withFallbacks(
            LLMProvider provider,
            List<LLMProvider> fallbacks) {
        return new ProviderSelection(provider, 100, null, fallbacks);
    }

    /**
     * Create a selection with full details
     */
    public static ProviderSelection full(
            LLMProvider provider,
            int score,
            String poolId,
            List<LLMProvider> fallbacks) {
        return new ProviderSelection(provider, score, poolId, fallbacks);
    }
}
