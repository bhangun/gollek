package tech.kayys.gollek.sdk.litert.config;

import lombok.Builder;
import lombok.Data;

/**
 * LiteRT configuration.
 */
@Data
@Builder
public class LiteRTConfig {

    /**
     * Number of CPU threads (0 = auto).
     */
    @Builder.Default
    private int numThreads = 4;

    /**
     * Delegate preference: NONE, CPU, GPU, NNAPI, COREML, AUTO.
     */
    @Builder.Default
    private Delegate delegate = Delegate.AUTO;

    /**
     * Enable XNNPACK optimization.
     */
    @Builder.Default
    private boolean enableXnnpack = true;

    /**
     * Use memory pool for faster allocations.
     */
    @Builder.Default
    private boolean useMemoryPool = true;

    /**
     * Memory pool size in bytes (0 = default 16MB).
     */
    @Builder.Default
    private long poolSizeBytes = 0;

    /**
     * Model cache directory (optional).
     */
    private String cacheDir;

    /**
     * Hardware delegate types.
     */
    public enum Delegate {
        NONE,
        CPU,
        GPU,
        NNAPI,
        COREML,
        AUTO
    }
}
