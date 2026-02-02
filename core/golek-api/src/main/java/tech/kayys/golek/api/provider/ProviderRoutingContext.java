package tech.kayys.golek.api.provider;

import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.wayang.tenant.TenantContext;

import java.time.Duration;
import java.util.Optional;

/**
 * Context for provider routing.
 */
public record ProviderRoutingContext(
        InferenceRequest request,
        TenantContext tenantContext,
        Optional<String> preferredProvider,
        Optional<String> deviceHint,
        Duration timeout,
        boolean costSensitive,
        int priority) {
}
