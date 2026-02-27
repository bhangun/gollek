package tech.kayys.gollek.inference.libtorch;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;

import java.util.Optional;

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
     * Default generation parameters.
     */
    GenerationConfig generation();

    // ── Nested config interfaces ──────────────────────────────────────

    interface NativeConfig {
        /**
         * Path to the LibTorch shared library directory.
         * If not set, defaults to GOLEK_LIBTORCH_LIB_PATH (CLI config) and then
         * loader fallbacks like LIBTORCH_PATH / ~/.gollek/source/vendor/libtorch.
         */
        Optional<String> libraryPath();
    }

    interface ModelConfig {
        /**
         * Base directory for TorchScript model files.
         */
        @WithDefault("${user.home}/.gollek/models/torchscript")
        String basePath();

        /**
         * Supported file extensions.
         */
        @WithDefault(".pt,.pts,.pth,.bin,.safetensors,.safetensor")
        String extensions();
    }

    interface GpuConfig {
        /**
         * Whether to use GPU (CUDA) if available.
         */
        @WithDefault("false")
        boolean enabled();

        /**
         * CUDA device index to use.
         */
        @WithDefault("0")
        int deviceIndex();
    }

    interface SessionConfig {
        /**
         * Maximum number of concurrent model sessions per tenant.
         */
        @WithDefault("4")
        int maxPerTenant();

        /**
         * Idle timeout for sessions in seconds.
         */
        @WithDefault("300")
        int idleTimeoutSeconds();

        /**
         * Maximum total sessions across all tenants.
         */
        @WithDefault("16")
        int maxTotal();
    }

    interface InferenceConfig {
        /**
         * Default inference timeout in seconds.
         */
        @WithDefault("30")
        int timeoutSeconds();

        /**
         * Number of threads for intra-op parallelism.
         */
        @WithDefault("4")
        int threads();
    }

    interface BatchingConfig {
        /**
         * Whether continuous batching is enabled.
         */
        @WithDefault("false")
        boolean enabled();

        /**
         * Maximum number of requests per batch.
         */
        @WithDefault("16")
        int maxBatchSize();

        /**
         * Maximum time in milliseconds to wait for a full batch before flushing.
         */
        @WithDefault("50")
        int batchTimeoutMs();
    }

    interface WarmupConfig {
        /**
         * Whether to preload models at startup.
         */
        @WithDefault("false")
        boolean enabled();

        /**
         * Comma-separated list of model IDs to preload at startup.
         * Example: "gpt2,bert-base"
         */
        Optional<String> models();

        /**
         * Whether to run a dummy forward pass after loading to trigger
         * JIT compilation and CUDA kernel caching.
         */
        @WithDefault("true")
        boolean dummyForward();

        /**
         * Tenant ID to use for warmup sessions.
         */
        @WithDefault("__warmup__")
        String tenantId();
    }

    interface GenerationConfig {
        /**
         * Default temperature for sampling (0.0 = deterministic).
         */
        @WithDefault("0.8")
        float temperature();

        /**
         * Default nucleus sampling probability.
         */
        @WithDefault("0.95")
        float topP();

        /**
         * Default top-k filtering.
         */
        @WithDefault("40")
        int topK();

        /**
         * Default maximum number of tokens to generate.
         */
        @WithDefault("512")
        int maxTokens();

        /**
         * Default repetition penalty.
         */
        @WithDefault("1.1")
        float repeatPenalty();

        /**
         * Default number of tokens to check for repetition.
         */
        @WithDefault("64")
        int repeatLastN();
    }
}
