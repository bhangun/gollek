package tech.kayys.gollek.inference.safetensor;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;

@ConfigMapping(prefix = "safetensor.provider")
public interface SafetensorProviderConfig {

    @WithDefault("true")
    boolean enabled();

    @WithDefault("${user.home}/.gollek/models/safetensors")
    String basePath();

    @WithDefault(".safetensors,.safetensor")
    String extensions();
}
