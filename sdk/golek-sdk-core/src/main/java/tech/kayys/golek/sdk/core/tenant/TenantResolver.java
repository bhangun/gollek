package tech.kayys.golek.sdk.core.tenant;

/**
 * Interface for resolving the current tenant ID.
 * Implementations can extract tenant from security context, headers, or other sources.
 */
public interface TenantResolver {
    
    /**
     * Resolves the current tenant ID.
     * 
     * @return The tenant ID, never null
     * @throws IllegalStateException if tenant cannot be resolved
     */
    String resolveTenantId();
    
    /**
     * Default implementation that returns a fixed tenant ID.
     */
    static TenantResolver fixed(String tenantId) {
        return () -> tenantId;
    }
}
