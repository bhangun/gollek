package tech.kayys.golek.provider.core.spi;

import java.time.Duration;

/**
 * Provider candidate with scoring
 */
public record ProviderCandidate(
                String providerId,
                LLMProvider provider,
                int score,
                Duration estimatedLatency,
                double estimatedCost) {
}