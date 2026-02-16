package tech.kayys.golek.spi.context;

import jakarta.ws.rs.core.SecurityContext;

/**
 * Interface for resolving RequestContext from SecurityContext.
 * Typically implemented by security modules or the runtime.
 */
public interface RequestContextResolver {

    /**
     * Resolves the request context from the provided security context.
     * 
     * @param securityContext The security context from the JAX-RS request
     * @return The resolved RequestContext
     */
    RequestContext resolve(SecurityContext securityContext);
}
