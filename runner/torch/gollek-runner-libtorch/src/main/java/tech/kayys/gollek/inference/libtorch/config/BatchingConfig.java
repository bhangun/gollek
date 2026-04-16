package tech.kayys.gollek.inference.libtorch.config;

import io.smallrye.config.WithDefault;

public interface BatchingConfig {
    /**
     * Whether continuous batching is enabled.
     */
    @WithDefault("false")
    boolean enabled();

    /**
     * Maximum number of requests per batch.
     */
    @WithDefault("16")
    int maxBatchSize();

    /**
     * Maximum time in milliseconds to wait for a full batch before flushing.
     */
    @WithDefault("50")
    int batchTimeoutMs();
}
