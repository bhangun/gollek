package tech.kayys.golek.spi.routing;

import tech.kayys.golek.spi.exception.ProviderException;
import tech.kayys.golek.spi.error.ErrorCode;

/**
 * Exception thrown when a provider's quota is exhausted.
 * Triggers failover to alternative providers.
 */
public class QuotaExhaustedException extends ProviderException {

    private final String tenantId;
    private final long quotaLimit;
    private final long currentUsage;

    public QuotaExhaustedException(String providerId, String tenantId) {
        super(providerId, String.format("Quota exhausted for provider '%s' (tenant: %s)",
                providerId, tenantId), null, ErrorCode.PROVIDER_QUOTA_EXCEEDED, false);
        this.tenantId = tenantId;
        this.quotaLimit = -1;
        this.currentUsage = -1;
    }

    public QuotaExhaustedException(
            String providerId,
            String tenantId,
            long quotaLimit,
            long currentUsage) {
        super(providerId, String.format(
                "Quota exhausted for provider '%s' (tenant: %s): %d/%d used",
                providerId, tenantId, currentUsage, quotaLimit), null, ErrorCode.PROVIDER_QUOTA_EXCEEDED, false);
        this.tenantId = tenantId;
        this.quotaLimit = quotaLimit;
        this.currentUsage = currentUsage;
    }

    // getProviderId() is inherited from ProviderException

    public String getTenantId() {
        return tenantId;
    }

    public long getQuotaLimit() {
        return quotaLimit;
    }

    public long getCurrentUsage() {
        return currentUsage;
    }

    /**
     * Check if detailed quota info is available
     */
    public boolean hasQuotaDetails() {
        return quotaLimit >= 0 && currentUsage >= 0;
    }
}
