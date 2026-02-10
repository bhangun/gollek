package tech.kayys.golek.plugin;

import tech.kayys.wayang.tenant.TenantId;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Default;

/**
 * Service for managing tenant quotas and enforcing limits.
 */
@Default
@ApplicationScoped
public interface TenantQuotaService {

    /**
     * Check current quota status for a tenant
     */
    QuotaInfo checkQuota(TenantId tenantId);

    /**
     * Reserve quota for a specific request
     */
    void reserve(TenantId tenantId, int amount);

    /**
     * Release previously reserved quota
     */
    void release(TenantId tenantId, int amount);

    /**
     * Get quota configuration for a tenant
     */
    QuotaConfig getConfig(TenantId tenantId);
}