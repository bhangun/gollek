package tech.kayys.golek.engine.tenant;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Repository for tenant-specific configurations
 */
@ApplicationScoped
public class TenantConfigRepository implements tech.kayys.wayang.tenant.TenantConfigurationService {

    private static final Logger LOG = Logger.getLogger(TenantConfigRepository.class);

    private final Map<String, TenantConfig> configs = new ConcurrentHashMap<>();

    @Override
    public Map<String, Object> getRunnerConfig(
            String tenantId,
            String runnerId) {
        TenantConfig config = configs.get(tenantId);

        if (config == null) {
            return Map.of();
        }

        return config.getRunnerConfig(runnerId);
    }

    public boolean isQuotaExhausted(String tenantId, String providerId) {
        TenantConfig config = configs.get(tenantId);

        if (config == null) {
            return false;
        }

        return config.isQuotaExhausted(providerId);
    }

    public boolean isCostSensitive(String tenantId) {
        TenantConfig config = configs.get(tenantId);

        if (config == null) {
            return false;
        }

        return config.isCostSensitive();
    }

    public void updateConfig(String tenantId, TenantConfig config) {
        configs.put(tenantId, config);
    }

    private static class TenantConfig {
        private final Map<String, Map<String, Object>> runnerConfigs;
        private final Map<String, QuotaInfo> quotas;
        private final boolean costSensitive;

        TenantConfig(
                Map<String, Map<String, Object>> runnerConfigs,
                Map<String, QuotaInfo> quotas,
                boolean costSensitive) {
            this.runnerConfigs = runnerConfigs;
            this.quotas = quotas;
            this.costSensitive = costSensitive;
        }

        Map<String, Object> getRunnerConfig(String runnerId) {
            return runnerConfigs.getOrDefault(runnerId, Map.of());
        }

        boolean isQuotaExhausted(String providerId) {
            QuotaInfo quota = quotas.get(providerId);
            return quota != null && quota.isExhausted();
        }

        boolean isCostSensitive() {
            return costSensitive;
        }
    }

    private static class QuotaInfo {
        private final long limit;
        private final long used;

        QuotaInfo(long limit, long used) {
            this.limit = limit;
            this.used = used;
        }

        boolean isExhausted() {
            return used >= limit;
        }
    }
}