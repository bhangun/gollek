package tech.kayys.golek.spi.provider;

import tech.kayys.golek.spi.inference.InferenceRequest;
// import tech.kayys.golek.spi.context.RequestContext; // Temporarily commented out due to missing dependency

import java.time.Duration;
import java.util.Optional;

/**
 * Context for provider routing.
 */
public record ProviderRoutingContext(
        InferenceRequest request,
        Object requestContext, // Using Object temporarily due to missing dependency
        Optional<String> preferredProvider,
        Optional<String> deviceHint,
        Duration timeout,
        boolean costSensitive,
        int priority) {
}
