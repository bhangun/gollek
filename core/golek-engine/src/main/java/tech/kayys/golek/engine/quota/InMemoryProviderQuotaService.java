package tech.kayys.golek.engine.quota;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.golek.provider.core.quota.ProviderQuotaService;

import org.jboss.logging.Logger;

import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * In-memory implementation of ProviderQuotaService.
 */
@ApplicationScoped
public class InMemoryProviderQuotaService implements ProviderQuotaService {

    private static final Logger LOG = Logger.getLogger(InMemoryProviderQuotaService.class);

    private final Map<String, AtomicLong> tokenUsage = new ConcurrentHashMap<>();
    private final Map<String, Instant> suspendedUntil = new ConcurrentHashMap<>();

    @Override
    public boolean hasQuota(String providerId) {
        // Check if provider is suspended (e.g. due to previous 429s)
        Instant suspension = suspendedUntil.get(providerId);
        if (suspension != null) {
            if (Instant.now().isBefore(suspension)) {
                return false;
            } else {
                suspendedUntil.remove(providerId);
            }
        }

        // In a real implementation, we would also check configured limits (e.g. monthly
        // budget)
        return true;
    }

    @Override
    public void recordUsage(String providerId, int tokensUsed) {
        tokenUsage.computeIfAbsent(providerId, k -> new AtomicLong(0))
                .addAndGet(tokensUsed);

        LOG.debugf("Recorded usage for provider %s: %d tokens (Total: %d)",
                providerId, tokensUsed, tokenUsage.get(providerId).get());
    }

    @Override
    public void reportExhaustion(String providerId, long retryAfterSeconds) {
        long delay = retryAfterSeconds > 0 ? retryAfterSeconds : 60; // Default 1 minute
        Instant resumeTime = Instant.now().plusSeconds(delay);
        suspendedUntil.put(providerId, resumeTime);

        LOG.warnf("Provider %s quota exhausted. Suspended until %s", providerId, resumeTime);
    }
}
