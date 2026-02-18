package tech.kayys.gollek.provider.core.quota;

import jakarta.enterprise.context.ApplicationScoped;
import io.quarkus.arc.DefaultBean;

@ApplicationScoped
@DefaultBean
public class DefaultProviderQuotaService implements ProviderQuotaService {

    @Override
    public boolean hasQuota(String providerId) {
        return true;
    }

    @Override
    public void recordUsage(String providerId, int tokensUsed) {
        // No-op
    }

    @Override
    public void reportExhaustion(String providerId, long retryAfterSeconds) {
        // No-op
    }
}
