package tech.kayys.golek.inference.djl;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;

@ConfigMapping(prefix = "djl.provider")
public interface DjlProviderConfig {

    @WithDefault("true")
    boolean enabled();

    @WithDefault("PyTorch")
    String engine();

    ModelConfig model();

    @WithDefault("120")
    int timeoutSeconds();

    interface ModelConfig {
        @WithDefault("${user.home}/.golek/models/djl")
        String basePath();

        @WithDefault(".pt,.pts,.jit,.ts")
        String extensions();
    }
}
