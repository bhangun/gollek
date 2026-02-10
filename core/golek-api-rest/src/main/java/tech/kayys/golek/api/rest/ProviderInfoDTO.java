package tech.kayys.golek.api.rest;

import tech.kayys.golek.spi.provider.ProviderCapabilities;

public record ProviderInfoDTO(
                String id,
                String name,
                ProviderCapabilities capabilities,
                boolean isHealthy,
                String healthMessage,
                String circuitBreakerState) {
}
