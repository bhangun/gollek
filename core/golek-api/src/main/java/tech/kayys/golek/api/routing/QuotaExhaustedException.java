package tech.kayys.golek.api.routing;

/**
 * Exception thrown when a provider's quota is exhausted.
 * Triggers failover to alternative providers.
 */
public class QuotaExhaustedException extends RuntimeException {

    private final String providerId;
    private final String tenantId;
    private final long quotaLimit;
    private final long currentUsage;

    public QuotaExhaustedException(String providerId, String tenantId) {
        super(String.format("Quota exhausted for provider '%s' (tenant: %s)", 
            providerId, tenantId));
        this.providerId = providerId;
        this.tenantId = tenantId;
        this.quotaLimit = -1;
        this.currentUsage = -1;
    }

    public QuotaExhaustedException(
            String providerId, 
            String tenantId, 
            long quotaLimit, 
            long currentUsage) {
        super(String.format(
            "Quota exhausted for provider '%s' (tenant: %s): %d/%d used", 
            providerId, tenantId, currentUsage, quotaLimit));
        this.providerId = providerId;
        this.tenantId = tenantId;
        this.quotaLimit = quotaLimit;
        this.currentUsage = currentUsage;
    }

    public String getProviderId() {
        return providerId;
    }

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
