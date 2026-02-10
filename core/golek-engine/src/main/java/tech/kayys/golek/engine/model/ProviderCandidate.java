package tech.kayys.golek.engine.model;

import java.time.Duration;

import tech.kayys.golek.spi.provider.LLMProvider;

public record ProviderCandidate(
        String providerId,
        LLMProvider provider,
        int score,
        Duration estimatedLatency,
        double estimatedCost) {
}