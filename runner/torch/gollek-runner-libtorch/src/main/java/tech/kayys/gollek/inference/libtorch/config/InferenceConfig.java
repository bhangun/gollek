package tech.kayys.gollek.inference.libtorch.config;

import io.smallrye.config.WithDefault;

public interface InferenceConfig {
    /**
     * Default inference timeout in seconds.
     */
    @WithDefault("30")
    int timeoutSeconds();

    /**
     * Number of threads for intra-op parallelism.
     */
    @WithDefault("4")
    int threads();
}
