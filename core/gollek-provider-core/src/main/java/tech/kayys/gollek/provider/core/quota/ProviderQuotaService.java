package tech.kayys.gollek.provider.core.quota;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Default;

/**
 * Service for tracking and enforcing provider-level quotas.
 * Used to rate-limit or cap usage on specific provider accounts.
 */
@Default
@ApplicationScoped
public interface ProviderQuotaService {

    /**
     * Check if a provider has remaining quota.
     * 
     * @param providerId The provider ID to check
     * @return true if quota is available, false otherwise
     */
    boolean hasQuota(String providerId);

    /**
     * Record usage for a provider.
     * 
     * @param providerId The provider ID
     * @param tokensUsed Number of tokens consumed
     */
    void recordUsage(String providerId, int tokensUsed);

    /**
     * Report a quota exhaustion event (e.g. 429 response).
     * 
     * @param providerId        The provider ID
     * @param retryAfterSeconds Optional retry delay
     */
    void reportExhaustion(String providerId, long retryAfterSeconds);
}
