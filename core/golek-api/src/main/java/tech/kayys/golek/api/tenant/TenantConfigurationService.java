package tech.kayys.golek.api.tenant;

import java.util.Map;

/**
 * Service for retrieving tenant-specific configuration.
 */
public interface TenantConfigurationService {

    /**
     * Get configuration for a specific runner and tenant.
     * 
     * @param tenantId The tenant ID
     * @param runnerId The runner ID
     * @return Configuration map
     */
    Map<String, Object> getRunnerConfig(String tenantId, String runnerId);
}
