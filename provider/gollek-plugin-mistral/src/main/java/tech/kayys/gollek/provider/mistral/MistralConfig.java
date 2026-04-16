package tech.kayys.gollek.provider.mistral;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

import java.time.Duration;

/**
 * Configuration for Mistral provider
 */
@ConfigMapping(prefix = "gollek.provider.mistral")
public interface MistralConfig {

    /**
     * Mistral API key
     */
    @WithName("api-key")
    String apiKey();

    /**
     * Default model to use if not specified
     */
    @WithName("default-model")
    @WithDefault("mistral-small-latest")
    String defaultModel();

    /**
     * Request timeout
     */
    @WithName("timeout")
    @WithDefault("PT60S")
    Duration timeout();

    /**
     * Enable/disable provider
     */
    @WithName("enabled")
    @WithDefault("true")
    boolean enabled();

    /**
     * Base URL for Mistral API
     */
    @WithName("base-url")
    @WithDefault("https://api.mistral.ai/v1")
    String baseUrl();
}
