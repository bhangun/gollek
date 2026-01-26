package tech.kayys.golek.provider.core.spi;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.provider.ProviderCapabilities;
import tech.kayys.golek.api.provider.ProviderConfig;
import tech.kayys.golek.api.provider.ProviderHealth;
import tech.kayys.golek.api.provider.ProviderMetadata;
import tech.kayys.golek.api.provider.ProviderMetrics;
import tech.kayys.golek.api.provider.ProviderRequest;
import tech.kayys.golek.api.tenant.TenantContext;

import tech.kayys.golek.provider.core.exception.ProviderException;

import java.util.Optional;

/**
 * Core SPI for all LLM providers.
 * Implementations must be thread-safe and support multi-tenancy.
 * Service Provider Interface for LLM backends.
 * All format adapters must implement this interface.
 */
public interface LLMProvider {

    /**
     * Unique provider identifier.
     * Format: namespace/name (e.g., "tech.kayys/gguf-provider")
     */
    String id();

    /**
     * Human-readable provider name
     */
    String name();

    /**
     * Provider version (semantic versioning)
     */
    default String version() {
        return "1.0.0";
    }

    /**
     * Provider metadata (version, capabilities, etc.)
     */
    ProviderMetadata metadata();

    /**
     * Get provider capabilities (features, limits, formats)
     */
    ProviderCapabilities capabilities();

    /**
     * Initialize provider with configuration.
     * Called once during provider registration.
     * 
     * @param config Provider-specific configuration
     * @throws ProviderInitializationException if initialization fails
     */
    void initialize(ProviderConfig config) throws ProviderInitializationException;

    /**
     * Check if provider supports the requested model.
     * 
     * @param modelId       Model identifier
     * @param tenantContext Tenant context for isolation
     * @return true if provider can handle this model
     */
    boolean supports(String modelId, TenantContext tenantContext);

    /**
     * Execute inference request (reactive).
     * This is the primary entry point for inference.
     * 
     * @param request Normalized inference request
     * @param context Tenant context
     * @return Uni with inference response
     */
    Uni<InferenceResponse> infer(
            ProviderRequest request,
            TenantContext context);

    /**
     * Execute inference request (blocking).
     * Default implementation subscribes and awaits the reactive version.
     * 
     * @param request Normalized inference request
     * @param context Tenant context
     * @return Inference response
     * @throws ProviderException if inference fails
     */
    default InferenceResponse inferBlocking(
            ProviderRequest request,
            TenantContext context) throws ProviderException {
        return infer(request, context)
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
     * Should verify:
     * - Provider is initialized
     * - Required resources are available
     * - Backend is reachable (for remote providers)
     * 
     * @return Health status with diagnostics
     */
    Uni<ProviderHealth> health();

    /**
     * Get current metrics/statistics.
     * 
     * @return Optional metrics snapshot
     */
    default Optional<ProviderMetrics> metrics() {
        return Optional.empty();
    }

    /**
     * Graceful shutdown.
     * Release resources, close connections, etc.
     */
    void shutdown();

    /**
     * Provider initialization exception
     */
    class ProviderInitializationException extends Exception {
        public ProviderInitializationException(String message) {
            super(message);
        }

        public ProviderInitializationException(String message, Throwable cause) {
            super(message, cause);
        }
    }

}