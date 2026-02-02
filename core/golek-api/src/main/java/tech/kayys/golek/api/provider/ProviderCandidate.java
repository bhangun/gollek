package tech.kayys.golek.api.provider;

import java.time.Duration;

/**
 * Candidate provider for a request with scoring metadata.
 */
public record ProviderCandidate(
        String providerId,
        LLMProvider provider,
        int score,
        Duration estimatedLatency,
        double estimatedCost) {

    public ProviderCandidate(LLMProvider provider, int priority, int weight) {
        this(provider.id(), provider, priority, Duration.ZERO, 0.0);
    }
}
