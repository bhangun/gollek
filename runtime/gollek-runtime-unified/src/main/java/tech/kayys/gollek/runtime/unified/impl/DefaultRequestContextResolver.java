package tech.kayys.gollek.runtime.unified.impl;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Default;
import jakarta.ws.rs.core.SecurityContext;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.context.RequestContextResolver;

@ApplicationScoped
@Default
public class DefaultRequestContextResolver implements RequestContextResolver {

    @Override
    public RequestContext resolve(SecurityContext securityContext) {
        // Return a default tenant context for the unified runtime
        return DefaultRequestContext.INSTANCE;
    }
}