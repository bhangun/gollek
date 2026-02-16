package tech.kayys.golek.provider.cerebras;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

import java.time.Duration;

/**
 * Configuration for Cerebras provider
 */
@ConfigMapping(prefix = "golek.provider.cerebras")
public interface CerebrasConfig {

    /**
     * Cerebras API key
     */
    @WithName("api-key")
    String apiKey();

    /**
     * Default model
     */
    @WithName("default-model")
    @WithDefault("llama3.1-70b")
    String defaultModel();

    /**
     * Request timeout
     */
    @WithName("timeout")
    @WithDefault("PT30S")
    Duration timeout();

    /**
     * Enable/disable provider
     */
    @WithName("enabled")
    @WithDefault("true")
    boolean enabled();

    /**
     * Base URL
     */
    @WithName("base-url")
    @WithDefault("https://api.cerebras.ai")
    String baseUrl();

    /**
     * Prefer Cerebras for Llama models
     */
    @WithName("prefer-for-llama")
    @WithDefault("false")
    boolean preferForLlama();
}