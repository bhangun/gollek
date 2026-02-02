package tech.kayys.golek.api.provider;

import tech.kayys.golek.api.context.EngineContext;
import tech.kayys.wayang.tenant.TenantContext;

import java.util.Optional;

/**
 * Context provided to providers during inference.
 */
public interface ProviderContext {

    /**
     * Get the tenant context for this request.
     */
    TenantContext getTenantContext();

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
