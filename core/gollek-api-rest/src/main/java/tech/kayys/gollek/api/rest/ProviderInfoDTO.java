package tech.kayys.gollek.api.rest;

import tech.kayys.gollek.spi.provider.ProviderCapabilities;

public record ProviderInfoDTO(
                String id,
                String name,
                ProviderCapabilities capabilities,
                boolean isHealthy,
                String healthMessage,
                String circuitBreakerState) {
}
