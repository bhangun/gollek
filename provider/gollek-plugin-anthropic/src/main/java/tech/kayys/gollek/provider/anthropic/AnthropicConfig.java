package tech.kayys.gollek.provider.anthropic;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

/**
 * Anthropic provider configuration.
 */
@ConfigMapping(prefix = "gollek.providers.anthropic")
public interface AnthropicConfig {

    /**
     * Enable/disable Anthropic provider.
     */
    @WithName("enabled")
    @WithDefault("true")
    boolean enabled();

    /**
     * Anthropic API key.
     */
    @WithName("api-key")
    @WithDefault("dummy")
    String apiKey();

    /**
     * Anthropic API base URL.
     */
    @WithName("base-url")
    @WithDefault("https://api.anthropic.com")
    String baseUrl();

    /**
     * Default model to use.
     */
    @WithName("default-model")
    @WithDefault("claude-3-sonnet-20240229")
    String defaultModel();

    /**
     * Request timeout in seconds.
     */
    @WithName("timeout-seconds")
    @WithDefault("30")
    int timeoutSeconds();

    /**
     * Maximum retries for failed requests.
     */
    @WithName("max-retries")
    @WithDefault("3")
    int maxRetries();
}
