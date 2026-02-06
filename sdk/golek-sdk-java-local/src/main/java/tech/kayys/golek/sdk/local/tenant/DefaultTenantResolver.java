package tech.kayys.golek.sdk.local.tenant;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.golek.sdk.core.tenant.TenantResolver;

/**
 * Default tenant resolver that returns a fixed "default" tenant ID.
 * This can be replaced with a custom implementation that extracts tenant from security context.
 */
@ApplicationScoped
public class DefaultTenantResolver implements TenantResolver {
    
    @Override
    public String resolveTenantId() {
        // In a real implementation, this would extract from:
        // - Security context (JWT claims)
        // - Thread local context
        // - Request headers
        // - Database lookup
        return "default";
    }
}
