package tech.kayys.gollek.inference.libtorch.config;

import io.smallrye.config.WithDefault;

public interface GenerationConfig {
    /**
     * Default temperature for sampling (0.0 = deterministic).
     */
    @WithDefault("0.8")
    float temperature();

    /**
     * Default nucleus sampling probability.
     */
    @WithDefault("0.95")
    float topP();

    /**
     * Default top-k filtering.
     */
    @WithDefault("40")
    int topK();

    /**
     * Default maximum number of tokens to generate.
     */
    @WithDefault("512")
    int maxTokens();

    /**
     * Default repetition penalty.
     */
    @WithDefault("1.1")
    float repeatPenalty();

    /**
     * Default number of tokens to check for repetition.
     */
    @WithDefault("64")
    int repeatLastN();
}
