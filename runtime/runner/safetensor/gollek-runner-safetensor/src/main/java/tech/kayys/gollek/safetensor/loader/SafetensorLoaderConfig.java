package tech.kayys.gollek.safetensor.loader;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

@ConfigMapping(prefix = "gollek.safetensor.loader")
public interface SafetensorLoaderConfig {

    @WithDefault("true")
    @WithName("prefer-mmap")
    boolean preferMmap();

    @WithDefault("8388608")
    @WithName("read-chunk-bytes")
    int readChunkBytes();

    Validation validation();

    interface Validation {
        @WithDefault("true")
        boolean strict();

        @WithDefault("true")
        @WithName("warn-on-empty-tensors")
        boolean warnOnEmptyTensors();

        @WithDefault("104857600")
        @WithName("max-header-bytes")
        long maxHeaderBytes();
    }

    Cache cache();

    interface Cache {
        @WithDefault("true")
        boolean enabled();

        @WithDefault("8")
        @WithName("max-size")
        int maxSize();

        @WithDefault("300")
        @WithName("ttl-seconds")
        long ttlSeconds();
    }

    @WithDefault("true")
    @WithName("log-load-summary")
    boolean logLoadSummary();
}
