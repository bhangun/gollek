package tech.kayys.gollek.inference.libtorch.config;

import io.smallrye.config.WithDefault;

public interface GpuConfig {
    /**
     * Whether to use GPU (CUDA) if available.
     */
    @WithDefault("false")
    boolean enabled();

    /**
     * Auto-enable Metal (MPS) on Apple Silicon when CUDA is disabled.
     */
    @io.smallrye.config.WithName("auto-mps-enabled")
    @WithDefault("true")
    boolean autoMpsEnabled();

    /**
     * CUDA device index to use.
     */
    @WithDefault("0")
    int deviceIndex();
}
