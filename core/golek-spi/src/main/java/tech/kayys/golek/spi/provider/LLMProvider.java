package tech.kayys.golek.spi.provider;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.spi.inference.InferenceResponse;

import tech.kayys.golek.spi.exception.ProviderException;

import java.util.Optional;

/**
 * Core SPI for all LLM providers.
 * Implementations must be thread-safe and support multi-tenancy.
 */
public interface LLMProvider {

    /**
     * Unique provider identifier.
     */
    String id();

    /**
     * Human-readable provider name
     */
    String name();

    /**
     * Provider version
     */
    default String version() {
        return "1.0.0";
    }

    /**
     * Provider metadata
     */
    ProviderMetadata metadata();

    /**
     * Provider capabilities
     */
    ProviderCapabilities capabilities();

    /**
     * Initialize provider with configuration.
     */
    void initialize(ProviderConfig config) throws ProviderException.ProviderInitializationException;

    /**
     * Check if provider supports the requested model.
     */
    boolean supports(String modelId, ProviderRequest request);

    /**
     * Execute inference request (reactive).
     */
    Uni<InferenceResponse> infer(ProviderRequest request);

    /**
     * Execute inference request (blocking).
     */
    default InferenceResponse inferBlocking(ProviderRequest request) throws ProviderException {
        return infer(request)
                .await()
                .atMost(java.time.Duration.ofSeconds(30));
    }

    /**
     * Check if provider supports streaming
     */
    default boolean supportsStreaming() {
        return this instanceof StreamingProvider;
    }

    /**
     * Check if provider is available
     */
    default Uni<Boolean> isAvailable() {
        return health().map(h -> h.status() == ProviderHealth.Status.HEALTHY);
    }

    /**
     * Health check for this provider.
     */
    Uni<ProviderHealth> health();

    /**
     * Get current metrics/statistics.
     */
    default Optional<ProviderMetrics> metrics() {
        return Optional.empty();
    }

    /**
     * Graceful shutdown.
     */
    void shutdown();
}
