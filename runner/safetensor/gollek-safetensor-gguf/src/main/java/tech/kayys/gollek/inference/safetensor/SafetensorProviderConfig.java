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

    /**
     * Backend selection for safetensors execution.
     * direct = use native Safetensor engine (FFM/zero-copy, origin format).
     * auto = prefer direct if available, else GGUF conversion.
     * gguf = force GGUF conversion.
     * libtorch = delegate to LibTorch (requires TorchScript models, not raw safetensors).
     */
    @WithDefault("direct")
    String backend();

    /**
     * Output directory for converted GGUF models.
     */
    @WithDefault("${user.home}/.gollek/models/gguf/converted")
    String ggufOutputDir();
}
