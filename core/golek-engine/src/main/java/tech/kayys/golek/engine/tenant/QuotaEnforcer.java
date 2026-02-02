package tech.kayys.golek.engine.tenant;

import io.quarkus.redis.datasource.RedisDataSource;
import io.quarkus.redis.datasource.keys.KeyCommands;
import io.quarkus.redis.datasource.string.StringCommands;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import lombok.extern.slf4j.Slf4j;

import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Quota enforcement service with Redis-backed rate limiting.
 * 
 * <p>
 * Supports multiple enforcement strategies:
 * <ul>
 * <li>Database-backed quotas (hourly/daily/monthly limits)</li>
 * <li>Redis-backed rate limiting (requests per second)</li>
 * <li>In-memory quotas (for development)</li>
 * </ul>
 * 
 * <p>
 * Thread-safe and distributed-safe (via Redis).
 * 
 * @author bhangun
 * @since 1.0.0
 */
@ApplicationScoped
@Slf4j
public class QuotaEnforcer {

    @Inject
    RedisDataSource redisDataSource;

    @ConfigProperty(name = "inference.rate-limiting.enabled", defaultValue = "true")
    boolean rateLimitingEnabled;

    @ConfigProperty(name = "inference.rate-limiting.default-requests-per-second", defaultValue = "100")
    int defaultRps;

    @ConfigProperty(name = "inference.multitenancy.quota-enforcement.strict-mode", defaultValue = "true")
    boolean strictMode;

    private StringCommands<String, String> stringCommands;
    private KeyCommands<String> keyCommands;

    // In-memory fallback (when Redis unavailable)
    private final ConcurrentHashMap<String, AtomicLong> inMemoryCounters = new ConcurrentHashMap<>();

    @jakarta.annotation.PostConstruct
    void init() {
        if (redisDataSource != null) {
            this.stringCommands = redisDataSource.string(String.class);
            this.keyCommands = redisDataSource.key();
        }
    }

    /**
     * Check if tenant has quota available and increment usage if so.
     * 
     * @param tenantId     Tenant UUID
     * @param resourceType Resource type (e.g., "requests", "storage_gb")
     * @param amount       Amount to consume
     * @return true if quota available and incremented, false otherwise
     */
    @Transactional
    public boolean checkAndIncrementQuota(UUID tenantId, String resourceType, long amount) {
        // 1. Check database quota (long-term limits)
        TenantQuota quota = TenantQuota.findByTenantAndResource(tenantId, resourceType);

        if (quota == null) {
            log.warn("No quota found for tenant={}, resource={}", tenantId, resourceType);
            return !strictMode; // Allow if not in strict mode
        }

        // 2. Check if quota available
        if (!quota.hasQuotaAvailable(amount)) {
            log.warn("Quota exceeded: tenant={}, resource={}, used={}, limit={}",
                    tenantId, resourceType, quota.quotaUsed, quota.quotaLimit);
            return false;
        }

        // 3. Increment usage in database
        quota.incrementUsage(amount);
        quota.persist();

        log.debug("Quota incremented: tenant={}, resource={}, amount={}, newUsage={}",
                tenantId, resourceType, amount, quota.quotaUsed);

        return true;
    }

    /**
     * Check rate limit using Redis sliding window.
     * 
     * @param tenantId          Tenant identifier
     * @param requestsPerSecond Allowed requests per second
     * @return true if rate limit not exceeded
     */
    public boolean checkRateLimit(String tenantId, int requestsPerSecond) {
        if (!rateLimitingEnabled) {
            return true;
        }

        String key = "ratelimit:" + tenantId + ":rps";

        try {
            // Use Redis INCR with TTL for sliding window
            String currentStr = stringCommands.get(key);
            long current = currentStr != null ? Long.parseLong(currentStr) : 0;

            if (current >= requestsPerSecond) {
                log.warn("Rate limit exceeded: tenant={}, current={}, limit={}",
                        tenantId, current, requestsPerSecond);
                return false;
            }

            // Increment and set TTL
            long newValue = stringCommands.incr(key);
            if (newValue == 1) {
                // First request in window - set TTL
                keyCommands.expire(key, Duration.ofSeconds(1));
            }

            return true;

        } catch (Exception e) {
            log.error("Redis rate limiting failed, falling back to in-memory", e);
            return checkRateLimitInMemory(tenantId, requestsPerSecond);
        }
    }

    /**
     * Fallback in-memory rate limiting (not distributed).
     */
    private boolean checkRateLimitInMemory(String tenantId, int requestsPerSecond) {
        String key = "ratelimit:" + tenantId;
        AtomicLong counter = inMemoryCounters.computeIfAbsent(key, k -> new AtomicLong(0));

        long current = counter.incrementAndGet();

        // Reset counter every second (approximate)
        if (current == 1) {
            resetCounterAsync(key, counter);
        }

        return current <= requestsPerSecond;
    }

    private void resetCounterAsync(String key, AtomicLong counter) {
        new Thread(() -> {
            try {
                Thread.sleep(1000);
                counter.set(0);
                inMemoryCounters.remove(key);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();
    }

    /**
     * Get current quota usage for a resource.
     */
    public long getCurrentUsage(UUID tenantId, String resourceType) {
        TenantQuota quota = TenantQuota.findByTenantAndResource(tenantId, resourceType);
        return quota != null ? quota.quotaUsed : 0;
    }

    /**
     * Get quota limit for a resource.
     */
    public long getQuotaLimit(UUID tenantId, String resourceType) {
        TenantQuota quota = TenantQuota.findByTenantAndResource(tenantId, resourceType);
        return quota != null ? quota.quotaLimit : 0;
    }

    /**
     * Check concurrent request limit.
     */
    public boolean checkConcurrentRequests(String tenantId, int maxConcurrent) {
        String key = "concurrent:" + tenantId;

        try {
            String currentStr = stringCommands.get(key);
            long current = currentStr != null ? Long.parseLong(currentStr) : 0;

            if (current >= maxConcurrent) {
                return false;
            }

            stringCommands.incr(key);
            return true;

        } catch (Exception e) {
            log.warn("Failed to check concurrent requests", e);
            return true; // Fail open
        }
    }

    /**
     * Release concurrent request slot.
     */
    public void releaseConcurrentSlot(String tenantId) {
        String key = "concurrent:" + tenantId;

        try {
            String currentStr = stringCommands.get(key);
            if (currentStr != null) {
                long current = Long.parseLong(currentStr);
                if (current > 0) {
                    stringCommands.decr(key);
                }
            }
        } catch (Exception e) {
            log.warn("Failed to release concurrent slot", e);
        }
    }

    /**
     * Reset quota usage (called by scheduled job).
     */
    @Transactional
    public void resetExpiredQuotas() {
        // This is called by a scheduled job
        // SQL function handles the logic: reset_expired_quotas()
        jakarta.persistence.EntityManager em = jakarta.persistence.Persistence
                .createEntityManagerFactory("default")
                .createEntityManager();

        em.createNativeQuery("SELECT reset_expired_quotas()").executeUpdate();

        log.info("Expired quotas reset completed");
    }

    /**
     * Token bucket algorithm for smooth rate limiting.
     */
    public boolean tryAcquireToken(String tenantId, int tokensPerSecond) {
        String tokenKey = "tokens:" + tenantId;
        String lastRefillKey = "tokens:lastrefill:" + tenantId;

        try {
            long now = System.currentTimeMillis();
            String lastRefillStr = stringCommands.get(lastRefillKey);
            long lastRefill = lastRefillStr != null ? Long.parseLong(lastRefillStr) : now;

            // Calculate tokens to add based on time elapsed
            long elapsedMs = now - lastRefill;
            double tokensToAdd = (elapsedMs / 1000.0) * tokensPerSecond;

            String currentTokensStr = stringCommands.get(tokenKey);
            double currentTokens = currentTokensStr != null
                    ? Double.parseDouble(currentTokensStr)
                    : tokensPerSecond;

            // Refill bucket (max = tokensPerSecond)
            double newTokens = Math.min(currentTokens + tokensToAdd, tokensPerSecond);

            if (newTokens >= 1.0) {
                // Consume 1 token
                stringCommands.set(tokenKey, String.valueOf(newTokens - 1.0));
                stringCommands.set(lastRefillKey, String.valueOf(now));
                return true;
            }

            return false;

        } catch (Exception e) {
            log.warn("Token bucket failed", e);
            return true; // Fail open
        }
    }

    /**
     * Get quota statistics for monitoring.
     */
    public QuotaStats getQuotaStats(UUID tenantId) {
        var quotas = TenantQuota.findByTenant(tenantId);

        return new QuotaStats(
                tenantId,
                quotas.stream()
                        .collect(java.util.stream.Collectors.toMap(
                                q -> q.resourceType,
                                QuotaUsage::fromQuota)));
    }

    public record QuotaStats(
            UUID tenantId,
            java.util.Map<String, QuotaUsage> quotas) {
    }

    public record QuotaUsage(
            String resourceType,
            long used,
            long limit,
            double usagePercent,
            String resetPeriod) {
        static QuotaUsage fromQuota(TenantQuota quota) {
            return new QuotaUsage(
                    quota.resourceType,
                    quota.quotaUsed,
                    quota.quotaLimit,
                    quota.getUsagePercentage(),
                    quota.resetPeriod.name());
        }
    }
}
