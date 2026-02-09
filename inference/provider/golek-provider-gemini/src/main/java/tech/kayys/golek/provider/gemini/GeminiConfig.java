package tech.kayys.golek.provider.gemini;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

import java.time.Duration;

/**
 * Configuration for Gemini provider
 */
@ConfigMapping(prefix = "golek.provider.gemini")
public interface GeminiConfig {

    /**
     * Gemini API key
     */
    @WithName("api-key")
    String apiKey();

    /**
     * Default model to use if not specified
     */
    @WithName("default-model")
    @WithDefault("gemini-1.5-flash")
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
     * Maximum output tokens
     */
    @WithName("max-output-tokens")
    @WithDefault("8192")
    int maxOutputTokens();

    /**
     * Safety settings threshold
     */
    @WithName("safety-threshold")
    @WithDefault("BLOCK_MEDIUM_AND_ABOVE")
    String safetyThreshold();
}
