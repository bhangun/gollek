package tech.kayys.golek.engine.model;

import tech.kayys.golek.core.provider.LLMProvider;

import java.time.Duration;

public record ProviderCandidate(
        String providerId,
        LLMProvider provider,
        int score,
        Duration estimatedLatency,
        double estimatedCost
) {
}