package tech.kayys.golek.spi.provider;

import tech.kayys.golek.spi.context.EngineContext;
// import tech.kayys.wayang.tenant.TenantContext; // Temporarily commented out due to missing dependency

import java.util.Optional;

/**
 * Context provided to providers during inference.
 */
public interface ProviderContext {

    /**
     * Get the tenant context for this request.
     */
    Object getTenantContext(); // Using Object temporarily due to missing dependency

    /**
     * Access to global engine services.
     */
    EngineContext getEngineContext();

    /**
     * Get a request-scoped attribute.
     */
    <T> Optional<T> getAttribute(String key, Class<T> type);

    /**
     * Set a request-scoped attribute.
     */
    void setAttribute(String key, Object value);
}
