package tech.kayys.golek.inference.infrastructure.security;

import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.container.ContainerRequestFilter;
import jakarta.ws.rs.ext.Provider;
import jakarta.inject.Inject;

/**
 * Extracts and validates tenant context from JWT claims.
 */
@Provider
@ApplicationScoped
public class TenantFilter implements ContainerRequestFilter {

    @Inject
    JsonWebToken jwt;

    @Inject
    TenantResolver tenantResolver;

    @Override
    public void filter(ContainerRequestContext requestContext) {
        String tenantId = jwt.getClaim("tenant_id");
        
        if (tenantId == null || tenantId.isBlank()) {
            requestContext.abortWith(
                Response.status(Response.Status.FORBIDDEN)
                    .entity("Missing tenant claim")
                    .build()
            );
            return;
        }

        TenantContext tenant = tenantResolver.resolve(
            TenantId.of(tenantId)
        );

        // Store in request context
        requestContext.setProperty("tenantContext", tenant);
    }
}