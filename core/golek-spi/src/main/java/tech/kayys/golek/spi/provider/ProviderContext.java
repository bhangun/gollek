package tech.kayys.golek.spi.provider;

import tech.kayys.golek.spi.context.EngineContext;
// import tech.kayys.golek.spi.context.RequestContext; // Temporarily commented out due to missing dependency

import java.util.Optional;

/**
 * Context provided to providers during inference.
 */
public interface ProviderContext {

    /**
     * Get the context for this request.
     */
    default Object getApiKeyContext() {
        return getRequestContext();
    }

    /**
     * @deprecated Use {@link #getApiKeyContext()}.
     */
    @Deprecated
    Object getRequestContext(); // Using Object temporarily due to missing dependency

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
