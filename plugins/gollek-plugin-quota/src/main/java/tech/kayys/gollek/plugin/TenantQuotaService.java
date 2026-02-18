package tech.kayys.gollek.plugin;

import tech.kayys.wayang.tenant.RequestId;

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
    QuotaInfo checkQuota(RequestId requestId);

    /**
     * Reserve quota for a specific request
     */
    void reserve(RequestId requestId, int amount);

    /**
     * Release previously reserved quota
     */
    void release(RequestId requestId, int amount);

    /**
     * Get quota configuration for a tenant
     */
    QuotaConfig getConfig(RequestId requestId);
}