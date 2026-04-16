package tech.kayys.gollek.safetensor;

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

    /**
     * Backend selection for safetensors execution.
     * direct = use native Safetensor engine (FFM/zero-copy, origin format).
     */
    @WithDefault("direct")
    String backend();

    /**
     * Output directory for converted GGUF models (if any).
     */
    @WithDefault("${user.home}/.gollek/models/gguf/converted")
    String ggufOutputDir();
}
