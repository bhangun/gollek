package tech.kayys.golek.engine.routing.strategy;

import tech.kayys.golek.api.provider.ProviderHealth;
import tech.kayys.golek.api.provider.RoutingContext;
import tech.kayys.golek.api.routing.RoutingConfig;
import tech.kayys.golek.api.provider.LLMProvider;

import java.util.Comparator;
import java.util.List;

/**
 * Scored selection: multi-factor scoring algorithm.
 * Considers health, cost, user preference, and capabilities.
 */
public class ScoredSelector implements ProviderSelector {

    @Override
    public ProviderSelection select(
            List<LLMProvider> candidates,
            RoutingContext context,
            RoutingConfig config) {

        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("No candidates available");
        }

        // Score all candidates
        List<ScoredProvider> scored = candidates.stream()
                .map(p -> new ScoredProvider(p, calculateScore(p, context, config)))
                .sorted(Comparator.comparingInt(ScoredProvider::score).reversed())
                .toList();

        ScoredProvider winner = scored.get(0);

        // Build fallback list (next highest scores)
        List<LLMProvider> fallbacks = scored.stream()
                .skip(1)
                .limit(2)
                .map(ScoredProvider::provider)
                .toList();

        return ProviderSelection.full(winner.provider(), winner.score(), null, fallbacks);
    }

    /**
     * Calculate score for a provider
     */
    private int calculateScore(LLMProvider provider, RoutingContext context, RoutingConfig config) {
        int score = 0;

        // 1. User preference bonus (+100)
        if (context.preferredProvider().isPresent() &&
                provider.id().equals(context.preferredProvider().get())) {
            score += 100;
        }

        // 2. Health status (+50 if healthy)
        ProviderHealth health = provider.health().await().atMost(java.time.Duration.ofMillis(500));
        if (health.isHealthy()) {
            score += 50;
        } else if (health.status() == ProviderHealth.Status.DEGRADED) {
            score += 25;
        }

        // 3. Cost sensitivity bonus for local providers (+30)
        if (context.costSensitive() && isLocalProvider(provider.id())) {
            score += 30;
        }

        // 4. Configuration weight bonus
        int weight = config.getWeight(provider.id());
        score += weight * 5;

        // 5. Prefer local if configured (+20)
        if (config.preferLocal() && isLocalProvider(provider.id())) {
            score += 20;
        }

        // 6. Priority adjustment
        score += context.priority();

        return Math.max(0, score);
    }

    /**
     * Check if provider is local
     */
    private boolean isLocalProvider(String providerId) {
        String id = providerId.toLowerCase();
        return id.contains("local") || id.contains("ollama") ||
                id.contains("vllm") || id.contains("gguf") ||
                id.contains("onnx") || id.contains("pytorch");
    }

    private record ScoredProvider(LLMProvider provider, int score) {
    }

    @Override
    public String description() {
        return "Multi-factor scoring considering health, cost, and preferences";
    }
}
