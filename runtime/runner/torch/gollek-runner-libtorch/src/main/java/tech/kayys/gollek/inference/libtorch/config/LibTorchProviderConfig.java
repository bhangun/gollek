package tech.kayys.gollek.inference.libtorch.config;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;

/**
 * Configuration for the LibTorch provider.
 * Properties are read from {@code application.properties} or
 * {@code application.yaml}
 * under the {@code libtorch.provider} prefix.
 */
@ConfigMapping(prefix = "libtorch.provider")
public interface LibTorchProviderConfig {

    /**
     * Whether the LibTorch provider is enabled.
     */
    @WithDefault("true")
    boolean enabled();

    /**
     * Native library configuration.
     */
    NativeConfig nativeLib();

    /**
     * Model configuration.
     */
    ModelConfig model();

    /**
     * GPU/CUDA configuration.
     */
    GpuConfig gpu();

    /**
     * Session pool configuration.
     */
    SessionConfig session();

    /**
     * Inference configuration.
     */
    InferenceConfig inference();

    /**
     * Continuous batching configuration.
     */
    BatchingConfig batching();

    /**
     * Warmup / model preloading configuration.
     */
    WarmupConfig warmup();

    /**
     * Adapter (PEFT) configuration.
     */
    AdapterConfig adapter();

    /**
     * Advanced CUDA optimization configuration (feature-flagged).
     */
    AdvancedConfig advanced();

    /**
     * Default generation parameters.
     */
    GenerationConfig generation();
}
