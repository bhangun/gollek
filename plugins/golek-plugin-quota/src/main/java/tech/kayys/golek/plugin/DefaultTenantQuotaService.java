package tech.kayys.golek.plugin;

import tech.kayys.wayang.tenant.TenantId;

import jakarta.enterprise.context.ApplicationScoped;
import java.time.Instant;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Default implementation of TenantQuotaService using in-memory storage.
 * This implementation is suitable for single-node deployments.
 */
@ApplicationScoped
public class DefaultTenantQuotaService implements TenantQuotaService {

    // Store quota usage per tenant
    private final ConcurrentHashMap<String, AtomicLong> usageMap = new ConcurrentHashMap<>();

    // Store quota configurations per tenant
    private final ConcurrentHashMap<String, QuotaConfig> configMap = new ConcurrentHashMap<>();

    // Store last reset time per tenant
    private final ConcurrentHashMap<String, Long> lastResetMap = new ConcurrentHashMap<>();

    @Override
    public QuotaInfo checkQuota(TenantId tenantId) {
        String tenantStr = tenantId.value();

        // Get or create default config
        QuotaConfig config = configMap.computeIfAbsent(tenantStr, this::getDefaultConfig);

        if (!config.isEnabled()) {
            // If quota is disabled, return unlimited quota info
            return new QuotaInfo("unlimited-" + tenantStr, 0, Long.MAX_VALUE, Long.MAX_VALUE,
                    Instant.now().toEpochMilli());
        }

        // Check if we need to reset the counter
        long currentTime = System.currentTimeMillis();
        long lastReset = lastResetMap.getOrDefault(tenantStr, 0L);

        if (currentTime - lastReset >= config.getPeriodMs()) {
            // Reset the counter
            usageMap.put(tenantStr, new AtomicLong(0));
            lastResetMap.put(tenantStr, currentTime);
        }

        AtomicLong currentUsage = usageMap.computeIfAbsent(tenantStr, k -> new AtomicLong(0));
        long current = currentUsage.get();
        long remaining = config.getLimit() - current;

        return new QuotaInfo(
                "quota-" + tenantStr,
                current,
                config.getLimit(),
                Math.max(0, remaining),
                lastReset + config.getPeriodMs());
    }

    @Override
    public void reserve(TenantId tenantId, int amount) {
        String tenantStr = tenantId.value();

        QuotaConfig config = configMap.computeIfAbsent(tenantStr, this::getDefaultConfig);

        if (!config.isEnabled()) {
            // If quota is disabled, no need to reserve
            return;
        }

        AtomicLong currentUsage = usageMap.computeIfAbsent(tenantStr, k -> new AtomicLong(0));
        currentUsage.addAndGet(amount);
    }

    @Override
    public void release(TenantId tenantId, int amount) {
        String tenantStr = tenantId.value();

        QuotaConfig config = configMap.computeIfAbsent(tenantStr, this::getDefaultConfig);

        if (!config.isEnabled()) {
            // If quota is disabled, no need to release
            return;
        }

        AtomicLong currentUsage = usageMap.computeIfAbsent(tenantStr, k -> new AtomicLong(0));
        currentUsage.addAndGet(-amount);

        // Ensure we don't go below 0
        long current = currentUsage.get();
        if (current < 0) {
            currentUsage.set(0);
        }
    }

    @Override
    public QuotaConfig getConfig(TenantId tenantId) {
        return configMap.computeIfAbsent(tenantId.value(), this::getDefaultConfig);
    }

    /**
     * Set quota configuration for a tenant
     */
    public void setConfig(TenantId tenantId, QuotaConfig config) {
        configMap.put(tenantId.value(), config);
    }

    /**
     * Get default configuration for a tenant
     */
    private QuotaConfig getDefaultConfig(String tenantId) {
        return QuotaConfig.builder()
                .limit(1000) // Default 1000 requests per hour
                .periodMs(3600000) // 1 hour
                .unit("requests")
                .enabled(true)
                .build();
    }
}