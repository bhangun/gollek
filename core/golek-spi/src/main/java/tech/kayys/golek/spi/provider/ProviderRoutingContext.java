package tech.kayys.golek.spi.provider;

import tech.kayys.golek.spi.inference.InferenceRequest;
// import tech.kayys.wayang.tenant.TenantContext; // Temporarily commented out due to missing dependency

import java.time.Duration;
import java.util.Optional;

/**
 * Context for provider routing.
 */
public record ProviderRoutingContext(
        InferenceRequest request,
        Object tenantContext, // Using Object temporarily due to missing dependency
        Optional<String> preferredProvider,
        Optional<String> deviceHint,
        Duration timeout,
        boolean costSensitive,
        int priority) {
}
