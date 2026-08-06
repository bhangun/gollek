package tech.kayys.gollek.core.config;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;

@ConfigMapping(prefix = "hardware")
public interface HardwareConfig {

    @WithDefault("false")
    boolean cudaEnabled();

    @WithDefault("false")
    boolean rocmEnabled();

    @WithDefault("false")
    boolean tpuEnabled();

    @WithDefault("false")
    boolean appleSiliconEnabled();

    @WithDefault("false")
    boolean openVINOEnabled();

    @WithDefault("8589934592")
    long availableMemory();
}
