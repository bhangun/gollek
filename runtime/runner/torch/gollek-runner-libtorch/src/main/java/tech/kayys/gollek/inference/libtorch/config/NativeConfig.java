package tech.kayys.gollek.inference.libtorch.config;

import java.util.Optional;

public interface NativeConfig {
    /**
     * Path to the LibTorch shared library directory.
     * If not set, defaults to GOLLEK_LIBTORCH_LIB_PATH (CLI config) and then
     * loader fallbacks like LIBTORCH_PATH / ~/.gollek/source/vendor/libtorch.
     */
    Optional<String> libraryPath();
}
