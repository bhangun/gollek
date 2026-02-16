package tech.kayys.golek.inference.transformers;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;

/**
 * Configuration for transformers runtime provider.
 */
@ConfigMapping(prefix = "transformers.provider")
public interface TransformersProviderConfig {

    @WithDefault("true")
    boolean enabled();

    @WithDefault("${user.home}/.golek/models/transformers")
    String basePath();

    @WithDefault("python3")
    String pythonCommand();

    @WithDefault("120")
    int timeoutSeconds();
}
