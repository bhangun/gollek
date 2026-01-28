package tech.kayys.golek.infrastructure;

import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.container.ContainerRequestFilter;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.ext.Provider;
import jakarta.inject.Inject;
import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.jwt.JsonWebToken;
import tech.kayys.golek.api.tenant.TenantContext;
import tech.kayys.golek.api.tenant.TenantId;
import tech.kayys.golek.api.tenant.TenantResolver;

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
                            .build());
            return;
        }

        TenantContext tenant = tenantResolver.resolve(
                new TenantId(tenantId));

        // Store in request context
        requestContext.setProperty("tenantContext", tenant);
    }
}