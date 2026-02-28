package tech.kayys.gollek.inference.libtorch;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;

import java.util.List;
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

    interface AdapterConfig {
        /**
         * Whether adapter-aware routing is enabled.
         */
        @WithDefault("true")
        boolean enabled();

        /**
         * Base directory for relative adapter paths.
         */
        @WithDefault("${user.home}/.gollek/models/torchscript/adapters")
        String basePath();

        /**
         * If true, adapter_path may point to a precompiled TorchScript model variant.
         * Runtime LoRA patching from safetensors is supported regardless of this flag.
         */
        @WithDefault("true")
        boolean allowPrecompiledModelPath();

        /**
         * Maximum unique adapter pools per tenant.
         * 0 = fallback to session.max-per-tenant.
         */
        @WithDefault("0")
        int maxActivePoolsPerTenant();

        /**
         * Enable rollout-guard policy checks for adapters.
         */
        @WithDefault("false")
        boolean rolloutGuardEnabled();

        /**
         * Optional allow-list of tenant IDs allowed to use adapters.
         * Empty = all tenants allowed.
         */
        Optional<List<String>> rolloutAllowedTenants();

        /**
         * Optional deny-list of adapter IDs blocked from serving.
         */
        Optional<List<String>> rolloutBlockedAdapterIds();

        /**
         * Optional deny-list of adapter path prefixes blocked from serving.
         */
        Optional<List<String>> rolloutBlockedPathPrefixes();
    }

    interface AdvancedConfig {
        /**
         * Master switch for advanced CUDA optimization path.
         */
        @WithDefault("false")
        boolean enabled();

        /**
         * Attention implementation mode.
         * Supported values: baseline, hybrid_fp8_bf16
         */
        @WithDefault("baseline")
        String attentionMode();

        /**
         * Enable FP8 row-wise quantized weight path.
         */
        @WithDefault("false")
        boolean fp8RowwiseEnabled();

        /**
         * Optional tenant allow-list for FP8 rowwise canary mode.
         * Empty = all tenants are eligible.
         */
        Optional<List<String>> fp8RowwiseAllowedTenants();

        /**
         * Optional model allow-list for FP8 rowwise canary mode.
         * Empty = all models are eligible.
         */
        Optional<List<String>> fp8RowwiseAllowedModels();

        /**
         * Optional tenant deny-list for FP8 rowwise canary mode.
         * Deny-list takes precedence over allow-list.
         */
        Optional<List<String>> fp8RowwiseBlockedTenants();

        /**
         * Optional model deny-list for FP8 rowwise canary mode.
         * Deny-list takes precedence over allow-list.
         */
        Optional<List<String>> fp8RowwiseBlockedModels();

        /**
         * Enable SageAttention2-like experimental path.
         */
        @WithDefault("false")
        boolean sageAttention2Enabled();

        /**
         * Optional tenant allow-list for SageAttention2 canary mode.
         * Empty = all tenants are eligible.
         */
        Optional<List<String>> sageAttention2AllowedTenants();

        /**
         * Optional model allow-list for SageAttention2 canary mode.
         * Empty = all models are eligible.
         */
        Optional<List<String>> sageAttention2AllowedModels();

        /**
         * Optional tenant deny-list for SageAttention2 canary mode.
         * Deny-list takes precedence over allow-list.
         */
        Optional<List<String>> sageAttention2BlockedTenants();

        /**
         * Optional model deny-list for SageAttention2 canary mode.
         * Deny-list takes precedence over allow-list.
         */
        Optional<List<String>> sageAttention2BlockedModels();

        /**
         * Allow-list of GPU SM versions for advanced path (comma-separated).
         * Example: "89,90"
         */
        @WithDefault("89,90")
        String allowedGpuSm();
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
