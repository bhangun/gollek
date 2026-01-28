package tech.kayys.golek.provider.embedding;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

@ConfigMapping(prefix = "golek.provider.embedding")
public interface EmbeddingConfig {

    /**
     * Enable/disable provider
     */
    @WithName("enabled")
    @WithDefault("true")
    boolean enabled();

    /**
     * Default model name
     */
    @WithName("default-model")
    @WithDefault("all-MiniLM-L6-v2")
    String defaultModel();

    /**
     * Path to model files
     */
    @WithName("model-path")
    @WithDefault("models")
    String modelPath();

    /**
     * Default embedding dimension
     */
    @WithName("dimension")
    @WithDefault("384")
    int defaultDimension();
}
