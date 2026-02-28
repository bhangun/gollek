package tech.kayys.gollek.inference.gguf;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

import java.time.Duration;
import java.util.List;
import java.util.Optional;

/**
 * Production configuration for GGUF provider using Quarkus ConfigMapping.
 * All settings are externalized and can be configured via
 * application.properties.
 */
@ConfigMapping(prefix = "gguf.provider")
public interface GGUFProviderConfig {

    /**
     * Enable GGUF provider
     */
    @WithName("enabled")
    @WithDefault("true")
    boolean enabled();

    /**
     * Enable verbose native debug logging (Metal/CUDA pipeline info)
     */
    @WithName("verbose-logging")
    @WithDefault("false")
    boolean verboseLogging();

    /**
     * Optional absolute path to the llama native library file (e.g.
     * /opt/llama/libllama.dylib).
     */
    @WithName("native.library-path")
    Optional<String> nativeLibraryPath();

    /**
     * Optional directory containing llama/ggml native libraries.
     */
    @WithName("native.library-dir")
    Optional<String> nativeLibraryDir();

    /**
     * Base directory for GGUF model files
     */
    @WithName("model.base-path")
    @WithDefault("${user.home}/.gollek/models/gguf")
    String modelBasePath();

    /**
     * Maximum context window size in tokens
     */
    @WithName("max-context-tokens")
    @WithDefault("4096")
    int maxContextTokens();

    /**
     * Enable GPU acceleration
     */
    @WithName("gpu.enabled")
    @WithDefault("false")
    boolean gpuEnabled();

    /**
     * Number of layers to offload to GPU (0 = CPU only, -1 = all layers)
     */
    @WithName("gpu.layers")
    @WithDefault("0")
    int gpuLayers();

    /**
     * GPU device ID to use
     */
    @WithName("gpu.device-id")
    @WithDefault("0")
    int gpuDeviceId();

    /**
     * Number of threads for CPU inference
     */
    @WithName("threads")
    @WithDefault("4")
    int threads();

    /**
     * Batch size for token processing
     */
    @WithName("batch-size")
    @WithDefault("128")
    int batchSize();

    /**
     * Enable memory mapping for model loading
     */
    @WithName("mmap.enabled")
    @WithDefault("true")
    boolean mmapEnabled();

    /**
     * Lock model in memory (prevents swapping)
     */
    @WithName("mlock.enabled")
    @WithDefault("false")
    boolean mlockEnabled();

    /**
     * Session pool minimum size per tenant/model combination
     */
    @WithName("session.pool.min-size")
    @WithDefault("1")
    int sessionPoolMinSize();

    /**
     * Session pool maximum size per tenant/model combination
     */
    @WithName("session.pool.max-size")
    @WithDefault("4")
    int sessionPoolMaxSize();

    /**
     * Session idle timeout before cleanup
     */
    @WithName("session.pool.idle-timeout")
    @WithDefault("PT5M")
    Duration sessionPoolIdleTimeout();

    /**
     * Convenience: session timeout in minutes
     */
    default long sessionTimeoutMinutes() {
        return sessionPoolIdleTimeout().toMinutes();
    }

    /**
     * Convenience: max sessions alias
     */
    default int maxSessions() {
        return sessionPoolMaxSize();
    }

    /**
     * Maximum concurrent inference requests
     */
    @WithName("max-concurrent-requests")
    @WithDefault("10")
    int maxConcurrentRequests();

    /**
     * Default inference timeout
     */
    @WithName("timeout")
    @WithDefault("PT30S")
    Duration defaultTimeout();

    /**
     * Circuit breaker failure threshold
     */
    @WithName("circuit-breaker.failure-threshold")
    @WithDefault("5")
    int circuitBreakerFailureThreshold();

    /**
     * Circuit breaker open duration
     */
    @WithName("circuit-breaker.open-duration")
    @WithDefault("PT30S")
    Duration circuitBreakerOpenDuration();

    /**
     * Circuit breaker half-open test requests
     */
    @WithName("circuit-breaker.half-open-permits")
    @WithDefault("3")
    int circuitBreakerHalfOpenPermits();

    /**
     * Circuit breaker half-open success threshold
     */
    @WithName("circuit-breaker.half-open-success-threshold")
    @WithDefault("2")
    int circuitBreakerHalfOpenSuccessThreshold();

    /**
     * Enable model prewarming on startup
     */
    @WithName("prewarm.enabled")
    @WithDefault("false")
    boolean prewarmEnabled();

    /**
     * List of model IDs to prewarm on startup
     */
    @WithName("prewarm.models")
    Optional<List<String>> prewarmModels();

    /**
     * Default temperature for sampling
     */
    @WithName("generation.temperature")
    @WithDefault("0.8")
    float defaultTemperature();

    /**
     * Default top-p for nucleus sampling
     */
    @WithName("generation.top-p")
    @WithDefault("0.95")
    float defaultTopP();

    /**
     * Default top-k for sampling
     */
    @WithName("generation.top-k")
    @WithDefault("40")
    int defaultTopK();

    /**
     * Default repeat penalty
     */
    @WithName("generation.repeat-penalty")
    @WithDefault("1.1")
    float defaultRepeatPenalty();

    /**
     * Default for JSON-only mode
     */
    @WithName("generation.json-mode")
    @WithDefault("false")
    boolean defaultJsonMode();

    /**
     * Default repeat last N tokens
     */
    @WithName("generation.repeat-last-n")
    @WithDefault("64")
    int defaultRepeatLastN();

    /**
     * Enable health checks
     */
    @WithName("health.enabled")
    @WithDefault("true")
    boolean healthEnabled();

    /**
     * Health check interval
     */
    @WithName("health.check-interval")
    @WithDefault("PT30S")
    Duration healthCheckInterval();

    /**
     * Maximum memory usage in bytes (0 = unlimited)
     */
    @WithName("memory.max-bytes")
    @WithDefault("0")
    long maxMemoryBytes();

    /**
     * Enable metrics collection
     */
    @WithName("metrics.enabled")
    @WithDefault("true")
    boolean metricsEnabled();

    /**
     * Enable LoRA adapter loading.
     */
    @WithName("lora.enabled")
    @WithDefault("true")
    boolean loraEnabled();

    /**
     * Base directory for relative LoRA adapter paths.
     */
    @WithName("lora.adapter-base-path")
    @WithDefault("${user.home}/.gollek/models/gguf/adapters")
    String loraAdapterBasePath();

    /**
     * Default adapter scale if request does not provide one.
     */
    @WithName("lora.default-scale")
    @WithDefault("1.0")
    float loraDefaultScale();

    /**
     * Soft bound for unique active adapters held by the manager cache.
     */
    @WithName("lora.max-active-adapters")
    @WithDefault("128")
    int loraMaxActiveAdapters();

    /**
     * Per-tenant cap for unique active adapters.
     */
    @WithName("lora.max-active-adapters-per-tenant")
    @WithDefault("16")
    int loraMaxActiveAdaptersPerTenant();

    /**
     * Enable rollout-guard policy checks for LoRA adapters.
     */
    @WithName("lora.rollout.guard-enabled")
    @WithDefault("false")
    boolean loraRolloutGuardEnabled();

    /**
     * Optional allow-list of tenant IDs allowed to use adapters.
     * Empty = all tenants allowed.
     */
    @WithName("lora.rollout.allowed-tenants")
    Optional<List<String>> loraRolloutAllowedTenants();

    /**
     * Optional deny-list of adapter IDs blocked from serving.
     */
    @WithName("lora.rollout.blocked-adapter-ids")
    Optional<List<String>> loraRolloutBlockedAdapterIds();

    /**
     * Optional deny-list of adapter path prefixes blocked from serving.
     */
    @WithName("lora.rollout.blocked-path-prefixes")
    Optional<List<String>> loraRolloutBlockedPathPrefixes();
}
