package tech.kayys.gollek.engine.config;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

import java.util.Optional;

/**
 * Central configuration for general model management.
 */
@ConfigMapping(prefix = "gollek.models")
public interface ModelConfig {

    /**
     * Enable or disable automatic model downloading.
     */
    @WithName("auto-download-enabled")
    @WithDefault("true")
    boolean autoDownloadEnabled();

    /**
     * Default model to use if none is specified in the request.
     */
    @WithName("default-model")
    @WithDefault("Qwen/Qwen2.5-0.5B-Instruct")
    Optional<String> defaultModel();

    /**
     * Default provider format to use (e.g., "gguf", "safetensors").
     */
    @WithName("default-provider")
    @WithDefault("gguf")
    String defaultProvider();
}
