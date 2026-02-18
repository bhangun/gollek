package tech.kayys.gollek.provider.ollama;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

import java.time.Duration;
import java.util.Optional;

/**
 * Configuration for Ollama provider
 */
@ConfigMapping(prefix = "gollek.provider.ollama")
public interface OllamaConfig {

    /**
     * Base URL of Ollama server
     */
    @WithName("base-url")
    @WithDefault("http://localhost:11434")
    String baseUrl();

    /**
     * Default model to use if not specified
     */
    @WithName("default-model")
    @WithDefault("llama3.1:8b")
    String defaultModel();

    /**
     * Request timeout
     */
    @WithName("timeout")
    @WithDefault("PT120S")
    Duration timeout();

    /**
     * Keep alive duration for model in memory
     */
    @WithName("keep-alive")
    @WithDefault("PT5M")
    Duration keepAlive();

    /**
     * Number of parallel requests allowed
     */
    @WithName("num-parallel")
    @WithDefault("4")
    int numParallel();

    /**
     * Optional GPU layers to offload
     */
    @WithName("num-gpu")
    Optional<Integer> numGpu();

    /**
     * Enable/disable provider
     */
    @WithName("enabled")
    @WithDefault("true")
    boolean enabled();
}
