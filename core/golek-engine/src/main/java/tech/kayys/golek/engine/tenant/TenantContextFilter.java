package tech.kayys.golek.engine.tenant;

import jakarta.annotation.Priority;
import jakarta.inject.Inject;
import jakarta.ws.rs.Priorities;
import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.container.ContainerRequestFilter;
import jakarta.ws.rs.ext.Provider;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.io.IOException;

/**
 * JAX-RS filter for tenant context initialization and validation.
 * 
 * <p>
 * Executed for every HTTP request before reaching the resource method.
 * 
 * <p>
 * Responsibilities:
 * <ul>
 * <li>Extract tenant ID from X-Tenant-ID header</li>
 * <li>Validate tenant exists and is active</li>
 * <li>Set tenant context for downstream processing</li>
 * <li>Add tenant info to MDC for logging</li>
 * </ul>
 * 
 * @author bhangun
 * @since 1.0.0
 */
@Provider
@Priority(Priorities.AUTHENTICATION + 1) // After authentication, before authorization
@Slf4j
public class TenantContextFilter implements ContainerRequestFilter {

    @ConfigProperty(name = "wayang.multitenancy.enabled", defaultValue = "false")
    boolean multitenancyEnabled;

    private static final String TENANT_HEADER = "X-Tenant-ID";
    private static final String REQUEST_ID_HEADER = "X-Request-ID";

    @Override
    public void filter(ContainerRequestContext requestContext) throws IOException {
        if (!multitenancyEnabled) {
            return;
        }

        String path = requestContext.getUriInfo().getPath();

        // Skip tenant validation for health checks and metrics
        if (isPublicEndpoint(path)) {
            return;
        }

        // Extract tenant ID from header
        String tenantId = requestContext.getHeaderString(TENANT_HEADER);

        if (tenantId == null || tenantId.trim().isEmpty()) {
            log.warn("Missing tenant ID header for path: {}", path);
            throw new AuthenticationException(
                    ErrorCode.AUTH_TENANT_NOT_FOUND,
                    "Tenant ID header (X-Tenant-ID) is required");
        }

        // Validate tenant
        Tenant tenant = Tenant.findByTenantId(tenantId);
        if (tenant == null) {
            log.warn("Unknown tenant: {}", tenantId);
            throw new AuthenticationException(
                    ErrorCode.AUTH_TENANT_NOT_FOUND,
                    "Tenant not found: " + tenantId);
        }

        if (tenant.status != Tenant.TenantStatus.ACTIVE) {
            log.warn("Inactive tenant attempted access: {}, status={}",
                    tenantId, tenant.status);
            throw new AuthenticationException(
                    ErrorCode.AUTH_TENANT_SUSPENDED,
                    "Tenant account is " + tenant.status);
        }

        // Set MDC for logging
        org.slf4j.MDC.put("tenantId", tenantId);

        // Also set request ID if present
        String requestId = requestContext.getHeaderString(REQUEST_ID_HEADER);
        if (requestId != null) {
            org.slf4j.MDC.put("requestId", requestId);
        }

        log.debug("Tenant context initialized: tenantId={}, path={}", tenantId, path);
    }

    private boolean isPublicEndpoint(String path) {
        return path.startsWith("q/") || // Quarkus endpoints (health, metrics)
                path.startsWith("openapi") || // OpenAPI spec
                path.equals("health") ||
                path.equals("metrics");
    }
}
