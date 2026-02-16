package tech.kayys.golek.engine.security;

import jakarta.annotation.Priority;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Alternative;
import io.smallrye.mutiny.Uni;
import java.util.UUID;

@Alternative
@Priority(1)
@ApplicationScoped
public class EngineQuotaEnforcer {

    public Uni<Boolean> checkAndIncrementQuota(UUID requestId, String resourceType, long amount) {
        // Default to allow in engine if redis is not present
        return Uni.createFrom().item(true);
    }

    public Uni<Boolean> checkRateLimit(String requestId, int rps) {
        return Uni.createFrom().item(true);
    }
}
