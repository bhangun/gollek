package tech.kayys.gollek.plugin;

// No RequestId import needed, using String

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
    QuotaInfo checkQuota(String requestId);

    /**
     * Reserve quota for a specific request
     */
    void reserve(String requestId, int amount);

    /**
     * Release previously reserved quota
     */
    void release(String requestId, int amount);

    /**
     * Get quota configuration for a tenant
     */
    QuotaConfig getConfig(String requestId);
}