package tech.kayys.gollek.inference.gguf;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.observability.AdapterMetricSchema;
import tech.kayys.gollek.spi.observability.AdapterMetricsRecorder;
import tech.kayys.gollek.spi.observability.AdapterSpec;
import tech.kayys.gollek.spi.observability.AdapterSpecResolver;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Resolves and validates LoRA adapter selections.
 */
@ApplicationScoped
public class LoraAdapterManager {

    private static final Logger log = Logger.getLogger(LoraAdapterManager.class);

    private final GGUFProviderConfig config;
    private final AdapterMetricsRecorder adapterMetricsRecorder;
    private final ConcurrentHashMap<String, AdapterSpec> resolved = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, String> adapterOwnerTenant = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Set<String>> tenantAdapterKeys = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Long> adapterLastAccessNanos = new ConcurrentHashMap<>();
    private final AtomicInteger activeAdapters = new AtomicInteger(0);

    @Inject
    public LoraAdapterManager(GGUFProviderConfig config, AdapterMetricsRecorder adapterMetricsRecorder) {
        this.config = config;
        this.adapterMetricsRecorder = adapterMetricsRecorder;
    }

    public Optional<AdapterSpec> resolve(ProviderRequest request) {
        return resolve(request, "community");
    }

    public Optional<AdapterSpec> resolve(ProviderRequest request, String tenantId) {
        if (!config.loraEnabled()) {
            return Optional.empty();
        }

        Optional<AdapterSpec> parsed = AdapterSpecResolver.fromProviderRequest(request, config.loraDefaultScale());
        if (parsed.isEmpty()) {
            return Optional.empty();
        }

        AdapterSpec raw = parsed.get();
        if (!raw.isType("lora")) {
            throw new IllegalArgumentException("GGUF provider only supports adapter_type=lora");
        }
        String resolvedPath = resolveAdapterPath(raw.adapterPath());
        if (resolvedPath == null) {
            throw new IllegalArgumentException("LoRA adapter path is required when LoRA is enabled");
        }

        if (!Files.exists(Path.of(resolvedPath))) {
            throw new IllegalArgumentException("LoRA adapter not found: " + resolvedPath);
        }

        AdapterSpec normalized = new AdapterSpec(
                raw.type(),
                raw.adapterId() == null ? resolvedPath : raw.adapterId(),
                resolvedPath,
                raw.scale());

        String normalizedTenant = normalizeTenantId(tenantId);
        enforceRolloutPolicy(normalizedTenant, normalized);
        String cacheKey = normalized.cacheKey();
        AdapterSpec existing = resolved.putIfAbsent(cacheKey, normalized);
        if (existing == null) {
            try {
                enforceTenantQuota(normalizedTenant, cacheKey);
            } catch (RuntimeException e) {
                resolved.remove(cacheKey, normalized);
                throw e;
            }
            adapterOwnerTenant.put(cacheKey, normalizedTenant);
            adapterLastAccessNanos.put(cacheKey, System.nanoTime());
            int count = activeAdapters.incrementAndGet();
            log.debugf("Registered LoRA adapter %s for tenant=%s (active=%d)",
                    normalized.adapterId(), normalizedTenant, count);
            recordAdapterCacheMetric("miss");
        } else {
            enforceTenantQuota(normalizedTenant, cacheKey);
            adapterLastAccessNanos.put(cacheKey, System.nanoTime());
            recordAdapterCacheMetric("hit");
        }

        enforceCapacity();
        return Optional.of(existing != null ? existing : normalized);
    }

    public int activeAdapterCount() {
        return activeAdapters.get();
    }

    private String resolveAdapterPath(String adapterPath) {
        if (adapterPath == null || adapterPath.isBlank()) {
            return null;
        }
        Path raw = Path.of(adapterPath);
        if (raw.isAbsolute()) {
            return raw.normalize().toString();
        }
        Path base = Path.of(config.loraAdapterBasePath());
        return base.resolve(raw).normalize().toString();
    }

    private void enforceCapacity() {
        int max = Math.max(1, config.loraMaxActiveAdapters());
        if (resolved.size() <= max) {
            return;
        }
        String key = findLeastRecentlyUsedAdapterKey();
        if (key != null && resolved.remove(key) != null) {
            String tenant = adapterOwnerTenant.remove(key);
            adapterLastAccessNanos.remove(key);
            if (tenant != null) {
                Set<String> keys = tenantAdapterKeys.get(tenant);
                if (keys != null) {
                    keys.remove(key);
                    if (keys.isEmpty()) {
                        tenantAdapterKeys.remove(tenant, keys);
                    }
                }
            }
            activeAdapters.decrementAndGet();
        }
    }

    private String findLeastRecentlyUsedAdapterKey() {
        String oldestKey = null;
        long oldestSeen = Long.MAX_VALUE;
        for (var entry : adapterLastAccessNanos.entrySet()) {
            if (entry.getValue() < oldestSeen) {
                oldestSeen = entry.getValue();
                oldestKey = entry.getKey();
            }
        }
        if (oldestKey != null) {
            return oldestKey;
        }
        return resolved.keys().hasMoreElements() ? resolved.keys().nextElement() : null;
    }

    private void enforceTenantQuota(String tenantId, String cacheKey) {
        Set<String> keys = tenantAdapterKeys.computeIfAbsent(tenantId, __ -> ConcurrentHashMap.newKeySet());
        if (keys.contains(cacheKey)) {
            return;
        }
        int maxPerTenant = perTenantAdapterLimit();
        if (keys.size() >= maxPerTenant) {
            throw new IllegalStateException(
                    "LoRA adapter quota exceeded for tenant '" + tenantId + "': max unique adapters="
                            + maxPerTenant + ". Increase GGUF session capacity or reuse adapter ids.");
        }
        keys.add(cacheKey);
    }

    private int perTenantAdapterLimit() {
        int configuredPerTenant = Math.max(1, config.loraMaxActiveAdaptersPerTenant());
        int bySessionCapacity = Math.max(1, config.sessionPoolMaxSize() * 4);
        int globalCap = Math.max(1, config.loraMaxActiveAdapters());
        int effectivePerTenant = Math.max(configuredPerTenant, bySessionCapacity);
        return Math.min(globalCap, effectivePerTenant);
    }

    private void enforceRolloutPolicy(String tenantId, AdapterSpec adapterSpec) {
        if (!config.loraRolloutGuardEnabled()) {
            return;
        }
        if (isBlockedTenant(tenantId)) {
            throw new IllegalStateException("Tenant '" + tenantId + "' is not allowed for LoRA rollout");
        }
        if (isBlockedAdapterId(adapterSpec.adapterId())) {
            throw new IllegalStateException(
                    "Adapter id '" + adapterSpec.adapterId() + "' is blocked by rollout policy");
        }
        if (isBlockedPath(adapterSpec.adapterPath())) {
            throw new IllegalStateException(
                    "Adapter path '" + adapterSpec.adapterPath() + "' is blocked by rollout policy");
        }
    }

    private boolean isBlockedTenant(String tenantId) {
        var allowed = config.loraRolloutAllowedTenants().orElse(java.util.List.of());
        if (allowed.isEmpty()) {
            return false;
        }
        return allowed.stream()
                .filter(v -> v != null && !v.isBlank())
                .noneMatch(v -> v.equalsIgnoreCase(tenantId));
    }

    private boolean isBlockedAdapterId(String adapterId) {
        if (adapterId == null || adapterId.isBlank()) {
            return false;
        }
        var blocked = config.loraRolloutBlockedAdapterIds().orElse(java.util.List.of());
        return blocked.stream()
                .filter(v -> v != null && !v.isBlank())
                .anyMatch(v -> v.equalsIgnoreCase(adapterId));
    }

    private boolean isBlockedPath(String adapterPath) {
        if (adapterPath == null || adapterPath.isBlank()) {
            return false;
        }
        String normalized = adapterPath.toLowerCase(Locale.ROOT);
        var blockedPrefixes = config.loraRolloutBlockedPathPrefixes().orElse(java.util.List.of());
        return blockedPrefixes.stream()
                .filter(v -> v != null && !v.isBlank())
                .map(v -> v.toLowerCase(Locale.ROOT))
                .anyMatch(normalized::startsWith);
    }

    private String normalizeTenantId(String tenantId) {
        if (tenantId == null || tenantId.isBlank()) {
            return "community";
        }
        return tenantId;
    }

    private void recordAdapterCacheMetric(String status) {
        if (!config.metricsEnabled() || adapterMetricsRecorder == null) {
            return;
        }
        var schema = AdapterMetricSchema.builder()
                .operation("adapter_cache_" + status)
                .build();
        adapterMetricsRecorder.recordSuccess(schema, 0L);
    }
}
