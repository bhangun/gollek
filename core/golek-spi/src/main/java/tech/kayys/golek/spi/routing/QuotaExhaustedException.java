package tech.kayys.golek.spi.routing;

import tech.kayys.golek.spi.exception.ProviderException;
import tech.kayys.golek.spi.error.ErrorCode;
import tech.kayys.golek.spi.auth.ApiKeyConstants;

/**
 * Exception thrown when a provider's quota is exhausted.
 * Triggers failover to alternative providers.
 */
public class QuotaExhaustedException extends ProviderException {

    private final String apiKey;
    private final long quotaLimit;
    private final long currentUsage;

    public QuotaExhaustedException(String providerId, String apiKey) {
        super(providerId, String.format("Quota exhausted for provider '%s' (tenant: %s)",
                providerId, apiKey), null, ErrorCode.PROVIDER_QUOTA_EXCEEDED, false);
        this.apiKey = apiKey;
        this.quotaLimit = -1;
        this.currentUsage = -1;
    }

    public QuotaExhaustedException(
            String providerId,
            String apiKey,
            long quotaLimit,
            long currentUsage) {
        super(providerId, String.format(
                "Quota exhausted for provider '%s' (tenant: %s): %d/%d used",
                providerId, apiKey, currentUsage, quotaLimit), null, ErrorCode.PROVIDER_QUOTA_EXCEEDED, false);
        this.apiKey = apiKey;
        this.quotaLimit = quotaLimit;
        this.currentUsage = currentUsage;
    }

    // getProviderId() is inherited from ProviderException

    public String getRequestId() {
        return apiKey;
    }

    public String getApiKey() {
        if (apiKey == null || apiKey.isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return apiKey;
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
