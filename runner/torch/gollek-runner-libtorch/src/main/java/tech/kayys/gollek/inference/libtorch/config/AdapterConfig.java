package tech.kayys.gollek.inference.libtorch.config;

import io.smallrye.config.WithDefault;

import java.util.List;
import java.util.Optional;

public interface AdapterConfig {
    /**
     * Whether adapter-aware routing is enabled.
     */
    @WithDefault("true")
    boolean enabled();

    /**
     * Base directory for relative adapter paths.
     */
    @WithDefault("${user.home}/.gollek/models/libtorchscript/adapters")
    String basePath();

    /**
     * If true, adapter_path may point to a precompiled TorchScript model variant.
     * Runtime LoRA patching from safetensors is supported regardless of this flag.
     */
    @WithDefault("true")
    boolean allowPrecompiledModelPath();

    /**
     * Maximum unique adapter pools per tenant.
     * 0 = fallback to session.max-per-tenant.
     */
    @WithDefault("0")
    int maxActivePoolsPerTenant();

    /**
     * Enable rollout-guard policy checks for adapters.
     */
    @WithDefault("false")
    boolean rolloutGuardEnabled();

    /**
     * Optional allow-list of tenant IDs allowed to use adapters.
     * Empty = all tenants allowed.
     */
    Optional<List<String>> rolloutAllowedTenants();

    /**
     * Optional deny-list of adapter IDs blocked from serving.
     */
    Optional<List<String>> rolloutBlockedAdapterIds();

    /**
     * Optional deny-list of adapter path prefixes blocked from serving.
     */
    Optional<List<String>> rolloutBlockedPathPrefixes();
}
