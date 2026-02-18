package tech.kayys.gollek.engine.routing.strategy;

import tech.kayys.gollek.spi.provider.RoutingContext;
import tech.kayys.gollek.spi.routing.RoutingConfig;
import tech.kayys.gollek.spi.provider.LLMProvider;

import java.util.Comparator;
import java.util.List;
import java.util.Set;

/**
 * Cost-optimized selection: prefers local/free providers over cloud.
 */
public class CostOptimizedSelector implements ProviderSelector {

    private static final Set<String> LOCAL_PREFIXES = Set.of(
            "local", "ollama", "onnx", "pytorch", "vllm", "gguf", "litert");

    private static final Set<String> EXPENSIVE_PREFIXES = Set.of(
            "openai", "anthropic", "gemini", "claude");

    @Override
    public ProviderSelection select(
            List<LLMProvider> candidates,
            RoutingContext context,
            RoutingConfig config) {

        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("No candidates available");
        }

        // Sort by cost score (higher = cheaper)
        List<LLMProvider> sorted = candidates.stream()
                .sorted(Comparator.comparingInt(this::getCostScore).reversed())
                .toList();

        LLMProvider selected = sorted.get(0);

        // Build fallback list
        List<LLMProvider> fallbacks = sorted.stream()
                .skip(1)
                .limit(2)
                .toList();

        return ProviderSelection.full(selected, getCostScore(selected), null, fallbacks);
    }

    /**
     * Calculate cost score (higher = cheaper)
     */
    private int getCostScore(LLMProvider provider) {
        String id = provider.id().toLowerCase();

        // Local providers are free
        for (String prefix : LOCAL_PREFIXES) {
            if (id.contains(prefix)) {
                return 100;
            }
        }

        // Cloud providers are expensive
        for (String prefix : EXPENSIVE_PREFIXES) {
            if (id.contains(prefix)) {
                return 20;
            }
        }

        // Unknown providers get medium score
        return 50;
    }

    @Override
    public String description() {
        return "Prefers local/free providers over cloud";
    }
}
