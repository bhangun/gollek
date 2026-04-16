package tech.kayys.gollek.inference.libtorch.config;

import io.smallrye.config.WithDefault;

public interface ModelConfig {
    /**
     * Base directory for TorchScript model files.
     */
    @WithDefault("${user.home}/.gollek/models/libtorchscript")
    String basePath();

    /**
     * Supported file extensions.
     */
    @WithDefault(".pt,.pts,.pth,.bin,.safetensors,.safetensor")
    String extensions();
}
