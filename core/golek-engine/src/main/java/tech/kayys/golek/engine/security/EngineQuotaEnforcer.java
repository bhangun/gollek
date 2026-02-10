package tech.kayys.golek.engine.security;

import jakarta.annotation.Priority;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Alternative;
import tech.kayys.wayang.tenant.QuotaEnforcer;
import io.smallrye.mutiny.Uni;
import java.util.UUID;

@Alternative
@Priority(1)
@ApplicationScoped
public class EngineQuotaEnforcer extends QuotaEnforcer {

    @Override
    public Uni<Boolean> checkAndIncrementQuota(UUID tenantId, String resourceType, long amount) {
        // Default to allow in engine if redis is not present
        return Uni.createFrom().item(true);
    }

    @Override
    public Uni<Boolean> checkRateLimit(String tenantId, int rps) {
        return Uni.createFrom().item(true);
    }
}
