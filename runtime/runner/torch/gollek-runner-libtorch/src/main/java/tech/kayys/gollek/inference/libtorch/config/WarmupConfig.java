package tech.kayys.gollek.inference.libtorch.config;

import io.smallrye.config.WithDefault;

import java.util.Optional;

public interface WarmupConfig {
    /**
     * Whether to preload models at startup.
     */
    @WithDefault("false")
    boolean enabled();

    /**
     * Comma-separated list of model IDs to preload at startup.
     * Example: "gpt2,bert-base"
     */
    Optional<String> models();

    /**
     * Whether to run a dummy forward pass after loading to trigger
     * JIT compilation and CUDA kernel caching.
     */
    @WithDefault("true")
    boolean dummyForward();

    /**
     * Tenant ID to use for warmup sessions.
     */
    @WithDefault("__warmup__")
    String tenantId();
}
