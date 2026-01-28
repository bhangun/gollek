package tech.kayys.golek.api.tenant;

/**
 * Resolves TenantContext from a TenantId.
 */
public interface TenantResolver {

    /**
     * Resolves the context for the given tenant ID.
     * 
     * @param tenantId the tenant ID to resolve
     * @return the resolved tenant context
     * @throws RuntimeException if the tenant cannot be resolved
     */
    TenantContext resolve(TenantId tenantId);
}
