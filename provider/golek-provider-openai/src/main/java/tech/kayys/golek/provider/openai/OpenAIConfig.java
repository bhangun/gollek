package tech.kayys.golek.provider.openai;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

import java.time.Duration;

/**
 * Configuration for OpenAI provider
 */
@ConfigMapping(prefix = "golek.provider.openai")
public interface OpenAIConfig {

    /**
     * OpenAI API key (Bearer token)
     */
    @WithName("api-key")
    String apiKey();

    /**
     * Default model
     */
    @WithName("default-model")
    @WithDefault("gpt-4o-mini")
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
     * Organization ID (optional)
     */
    @WithName("organization")
    java.util.Optional<String> organization();

    /**
     * Base URL (for Azure OpenAI or proxies)
     */
    @WithName("base-url")
    @WithDefault("https://api.openai.com")
    String baseUrl();
}