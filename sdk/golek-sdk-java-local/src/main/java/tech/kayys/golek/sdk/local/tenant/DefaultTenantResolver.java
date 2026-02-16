package tech.kayys.golek.sdk.local.tenant;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.golek.sdk.core.tenant.TenantResolver;

/**
 * Default tenant resolver that returns a fixed "default" tenant ID.
 * This can be replaced with a custom implementation that extracts tenant from
 * security context.
 */
@ApplicationScoped
public class DefaultTenantResolver implements TenantResolver {

    @Override
    public String resolveRequestId() {
        return "default";
    }

    @Override
    public String resolveApiKey() {
        return "default";
    }
}
