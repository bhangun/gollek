package tech.kayys.gollek.provider.litert;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

import java.time.Duration;

@ConfigMapping(prefix = "litert.provider")
public interface LiteRTProviderConfig {

    @WithName("model.base-path")
    @WithDefault("/var/lib/gollek/models/litert")
    String modelBasePath();

    @WithName("threads")
    @WithDefault("4")
    int threads();

    @WithName("gpu.enabled")
    @WithDefault("false")
    boolean gpuEnabled();

    @WithName("npu.enabled")
    @WithDefault("false")
    boolean npuEnabled();

    @WithName("gpu.backend")
    @WithDefault("auto")
    String gpuBackend();

    @WithName("npu.type")
    @WithDefault("auto")
    String npuType();

    @WithName("timeout")
    @WithDefault("PT30S")
    Duration defaultTimeout();
}
