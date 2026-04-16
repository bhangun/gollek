package tech.kayys.gollek.inference.libtorch.config;

import io.smallrye.config.WithDefault;

public interface SessionConfig {
    /**
     * Maximum number of concurrent model sessions per tenant.
     */
    @WithDefault("4")
    int maxPerTenant();

    /**
     * Idle timeout for sessions in seconds.
     */
    @WithDefault("300")
    int idleTimeoutSeconds();

    /**
     * Maximum total sessions across all tenants.
     */
    @WithDefault("16")
    int maxTotal();
}
