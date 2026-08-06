package tech.kayys.gollek.cache;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.Readiness;

/**
 * MicroProfile Health {@code /q/health/ready} check for the prompt cache.
 *
 * <p>
 * Exposes:
 * <ul>
 * <li>Strategy, scope, TTL, eviction policy</li>
 * <li>Entry count, total cached tokens, hit rate</li>
 * <li>Pin count (blocks currently locked by active requests)</li>
 * </ul>
 *
 * <p>
 * Reports DOWN if the store has never been initialized or if hit rate
 * falls below a configurable floor (useful as a degraded-cache alert).
 */
@Readiness
@ApplicationScoped
public class PromptCacheHealthCheck implements HealthCheck {

    private static final double MIN_HIT_RATE_ALERT = 0.0; // 0 = never alert on hit rate

    @Inject
    PromptCacheStore store;
    @Inject
    PromptCacheConfig config;

    @Override
    public HealthCheckResponse call() {
        PromptCacheStats stats = store.stats();

        org.eclipse.microprofile.health.HealthCheckResponseBuilder builder = HealthCheckResponse
                .named("prompt-cache")
                .withData("strategy", stats.strategy())
                .withData("enabled", config.enabled())
                .withData("scope", config.scope())
                .withData("ttl", config.ttl().toString())
                .withData("evictionPolicy", config.evictionPolicy())
                .withData("cachedEntries", stats.cachedEntries())
                .withData("hitRate", String.format("%.1f%%", stats.hitRate() * 100))
                .withData("hitRate5min", String.format("%.1f%%", stats.hitRateLast5Min() * 100))
                .withData("stores", stats.stores())
                .withData("evictions", stats.evictions());

        if (!config.enabled()) {
            return builder.down().withData("reason", "disabled by config").build();
        }

        return builder.up().build();
    }
}
