package tech.kayys.gollek.engine.model;

import java.time.Duration;

import tech.kayys.gollek.spi.provider.LLMProvider;

public record ProviderCandidate(
        String providerId,
        LLMProvider provider,
        int score,
        Duration estimatedLatency,
        double estimatedCost) {
}