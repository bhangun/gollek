Pecel Output
Generated: 2026-01-19 14:59:00
Files: 33 | Directories: 12 | Total Size: 118.3 KB


================================================================================
tech/kayys/golek/provider/adapter/AbstractProvider.java
Size: 9.0 KB | Modified: 2026-01-19 14:05:34
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.adapter;

import io.smallrye.mutiny.Uni;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;
import tech.kayys.wayang.inference.api.TenantContext;
import tech.kayys.wayang.inference.providers.circuit.CircuitBreaker;
import tech.kayys.wayang.inference.providers.circuit.DefaultCircuitBreaker;
import tech.kayys.wayang.inference.providers.ratelimit.RateLimiter;
import tech.kayys.wayang.inference.providers.ratelimit.TokenBucketRateLimiter;
import tech.kayys.wayang.inference.providers.spi.*;

import java.time.Duration;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Base implementation providing common functionality for all providers.
 * Handles initialization, health checks, rate limiting, and circuit breaking.
 */
public abstract class AbstractProvider implements LLMProvider {

    protected final Logger log = Logger.getLogger(getClass());

    private final AtomicBoolean initialized = new AtomicBoolean(false);
    private final AtomicReference<ProviderHealth> healthCache = new AtomicReference<>();
    private final Map<String, Object> configuration = new ConcurrentHashMap<>();
    private final Map<String, RateLimiter> rateLimiters = new ConcurrentHashMap<>();
    private final Map<String, CircuitBreaker> circuitBreakers = new ConcurrentHashMap<>();

    @ConfigProperty(name = "provider.health.cache.duration", defaultValue = "PT30S")
    protected Duration healthCacheDuration;

    @ConfigProperty(name = "provider.circuit-breaker.failure-threshold", defaultValue = "5")
    protected int circuitBreakerFailureThreshold;

    @ConfigProperty(name = "provider.circuit-breaker.timeout", defaultValue = "PT60S")
    protected Duration circuitBreakerTimeout;

    @ConfigProperty(name = "provider.rate-limit.enabled", defaultValue = "true")
    protected boolean rateLimitEnabled;

    @Override
    public final Uni<Void> initialize(Map<String, Object> config, TenantContext tenant) {
        if (initialized.get()) {
            log.warnf("Provider %s already initialized", providerId());
            return Uni.createFrom().voidItem();
        }

        log.infof("Initializing provider %s for tenant %s", 
            providerId(), tenant.getTenantId());

        this.configuration.putAll(config);

        return doInitialize(config, tenant)
            .invoke(() -> {
                initialized.set(true);
                log.infof("Provider %s initialized successfully", providerId());
            })
            .onFailure().invoke(ex -> 
                log.errorf(ex, "Failed to initialize provider %s", providerId())
            );
    }

    @Override
    public final Uni<ProviderResponse> infer(ProviderRequest request) {
        if (!initialized.get()) {
            return Uni.createFrom().failure(
                new ProviderException(providerId(), "Provider not initialized")
            );
        }

        String tenantId = request.getTenantContext() != null 
            ? request.getTenantContext().getTenantId() 
            : "default";

        return checkRateLimit(tenantId)
            .chain(() -> executeWithCircuitBreaker(request, tenantId))
            .onFailure().transform(this::handleFailure);
    }

    @Override
    public final Uni<ProviderHealth> health() {
        ProviderHealth cached = healthCache.get();
        if (cached != null && !isHealthCacheExpired(cached)) {
            return Uni.createFrom().item(cached);
        }

        return doHealthCheck()
            .invoke(healthCache::set)
            .onFailure().recoverWithItem(ex -> {
                log.warnf(ex, "Health check failed for provider %s", providerId());
                return ProviderHealth.unhealthy(ex.getMessage());
            });
    }

    @Override
    public final Uni<Void> shutdown() {
        if (!initialized.get()) {
            return Uni.createFrom().voidItem();
        }

        log.infof("Shutting down provider %s", providerId());

        return doShutdown()
            .invoke(() -> {
                initialized.set(false);
                configuration.clear();
                rateLimiters.clear();
                circuitBreakers.clear();
                log.infof("Provider %s shut down successfully", providerId());
            })
            .onFailure().invoke(ex ->
                log.errorf(ex, "Error during provider %s shutdown", providerId())
            );
    }

    /**
     * Provider-specific initialization logic
     */
    protected abstract Uni<Void> doInitialize(
        Map<String, Object> config, 
        TenantContext tenant
    );

    /**
     * Provider-specific inference logic
     */
    protected abstract Uni<ProviderResponse> doInfer(ProviderRequest request);

    /**
     * Provider-specific health check
     */
    protected abstract Uni<ProviderHealth> doHealthCheck();

    /**
     * Provider-specific shutdown logic
     */
    protected Uni<Void> doShutdown() {
        return Uni.createFrom().voidItem();
    }

    /**
     * Rate limiting check
     */
    protected Uni<Void> checkRateLimit(String tenantId) {
        if (!rateLimitEnabled) {
            return Uni.createFrom().voidItem();
        }

        RateLimiter rateLimiter = rateLimiters.computeIfAbsent(
            tenantId,
            id -> createRateLimiter(tenantId)
        );

        return Uni.createFrom().item(() -> {
            if (!rateLimiter.tryAcquire()) {
                throw new ProviderException(
                    providerId(),
                    "Rate limit exceeded for tenant: " + tenantId,
                    null,
                    true
                );
            }
            return null;
        });
    }

    /**
     * Execute with circuit breaker protection
     */
    protected Uni<ProviderResponse> executeWithCircuitBreaker(
        ProviderRequest request,
        String tenantId
    ) {
        CircuitBreaker circuitBreaker = circuitBreakers.computeIfAbsent(
            tenantId,
            id -> createCircuitBreaker(tenantId)
        );

        return circuitBreaker.call(() -> doInfer(request));
    }

    /**
     * Create rate limiter for tenant
     */
    protected RateLimiter createRateLimiter(String tenantId) {
        int requestsPerSecond = getConfigValue("rate-limit.requests-per-second", 10);
        int burst = getConfigValue("rate-limit.burst", 20);
        
        log.debugf("Creating rate limiter for tenant %s: %d req/s, burst %d",
            tenantId, requestsPerSecond, burst);
        
        return new TokenBucketRateLimiter(requestsPerSecond, burst);
    }

    /**
     * Create circuit breaker for tenant
     */
    protected CircuitBreaker createCircuitBreaker(String tenantId) {
        log.debugf("Creating circuit breaker for tenant %s: threshold %d, timeout %s",
            tenantId, circuitBreakerFailureThreshold, circuitBreakerTimeout);
        
        return new DefaultCircuitBreaker(
            circuitBreakerFailureThreshold,
            circuitBreakerTimeout
        );
    }

    /**
     * Handle execution failures
     */
    protected Throwable handleFailure(Throwable ex) {
        if (ex instanceof ProviderException) {
            return ex;
        }
        
        log.errorf(ex, "Provider %s execution failed", providerId());
        
        return new ProviderException(
            providerId(),
            "Provider execution failed: " + ex.getMessage(),
            ex,
            isRetryableError(ex)
        );
    }

    /**
     * Determine if error is retryable
     */
    protected boolean isRetryableError(Throwable ex) {
        String message = ex.getMessage() != null ? ex.getMessage().toLowerCase() : "";
        return message.contains("timeout") ||
               message.contains("connection") ||
               message.contains("network") ||
               message.contains("unavailable");
    }

    /**
     * Check if health cache is expired
     */
    private boolean isHealthCacheExpired(ProviderHealth health) {
        return health.timestamp()
            .plus(healthCacheDuration)
            .isBefore(java.time.Instant.now());
    }

    /**
     * Get configuration value with default
     */
    @SuppressWarnings("unchecked")
    protected <T> T getConfigValue(String key, T defaultValue) {
        return (T) configuration.getOrDefault(key, defaultValue);
    }

    /**
     * Get optional configuration value
     */
    @SuppressWarnings("unchecked")
    protected <T> Optional<T> getConfigValue(String key, Class<T> type) {
        Object value = configuration.get(key);
        if (value != null && type.isInstance(value)) {
            return Optional.of((T) value);
        }
        return Optional.empty();
    }

    /**
     * Check if provider is initialized
     */
    protected boolean isInitialized() {
        return initialized.get();
    }

    /**
     * Get provider configuration
     */
    protected Map<String, Object> getConfiguration() {
        return Map.copyOf(configuration);
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/adapter/CloudProviderAdapter.java
Size: 5.8 KB | Modified: 2026-01-19 14:06:19
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.adapter;

import io.smallrye.mutiny.Uni;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import tech.kayys.wayang.inference.api.TenantContext;
import tech.kayys.wayang.inference.providers.ratelimit.RateLimiter;
import tech.kayys.wayang.inference.providers.ratelimit.SlidingWindowRateLimiter;
import tech.kayys.wayang.inference.providers.spi.ProviderCapabilities;
import tech.kayys.wayang.inference.providers.spi.ProviderHealth;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Base adapter for cloud-based LLM providers (OpenAI, Anthropic, etc.)
 * Handles API authentication, rate limiting, and retry logic.
 */
public abstract class CloudProviderAdapter extends AbstractProvider {

    protected String apiKey;
    protected String baseUrl;
    protected Duration requestTimeout;
    protected final AtomicLong totalRequests = new AtomicLong();
    protected final AtomicLong totalErrors = new AtomicLong();

    @Override
    protected Uni<Void> doInitialize(Map<String, Object> config, TenantContext tenant) {
        return Uni.createFrom().item(() -> {
            this.apiKey = extractApiKey(config, tenant);
            this.baseUrl = getConfigValue("base-url", getDefaultBaseUrl());
            this.requestTimeout = Duration.parse(
                getConfigValue("request-timeout", "PT30S")
            );

            if (apiKey == null || apiKey.isBlank()) {
                throw new IllegalStateException(
                    "API key not configured for provider " + providerId()
                );
            }

            log.infof("Cloud provider %s initialized with base URL: %s",
                providerId(), baseUrl);

            return null;
        });
    }

    @Override
    protected Uni<ProviderHealth> doHealthCheck() {
        return performHealthCheckRequest()
            .map(success -> {
                if (success) {
                    return ProviderHealth.builder()
                        .status(ProviderHealth.Status.HEALTHY)
                        .message("Cloud provider API reachable")
                        .detail("base_url", baseUrl)
                        .detail("total_requests", totalRequests.get())
                        .detail("total_errors", totalErrors.get())
                        .build();
                } else {
                    return ProviderHealth.degraded("API health check failed");
                }
            })
            .onFailure().recoverWithItem(ex ->
                ProviderHealth.unhealthy("API unreachable: " + ex.getMessage())
            );
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
            .streaming(true)
            .functionCalling(supportsFunctionCalling())
            .multimodal(supportsMultimodal())
            .embeddings(supportsEmbeddings())
            .maxContextTokens(getMaxContextTokens())
            .maxOutputTokens(getMaxOutputTokens())
            .build();
    }

    @Override
    protected RateLimiter createRateLimiter(String tenantId) {
        // Cloud providers typically have stricter rate limits
        int requestsPerMinute = getConfigValue("rate-limit.requests-per-minute", 60);
        
        log.debugf("Creating sliding window rate limiter for tenant %s: %d req/min",
            tenantId, requestsPerMinute);
        
        return new SlidingWindowRateLimiter(requestsPerMinute, Duration.ofMinutes(1));
    }

    /**
     * Extract API key from config or tenant context
     */
    protected String extractApiKey(Map<String, Object> config, TenantContext tenant) {
        // Try config first
        String key = (String) config.get("api-key");
        
        // Try environment variable
        if (key == null || key.isBlank()) {
            String envVar = getApiKeyEnvironmentVariable();
            key = System.getenv(envVar);
        }
        
        // Try tenant-specific key
        if (key == null || key.isBlank()) {
            key = tenant.getAttribute("api-key-" + providerId()).orElse(null);
        }
        
        return key;
    }

    /**
     * Perform health check request to provider API
     */
    protected abstract Uni<Boolean> performHealthCheckRequest();

    /**
     * Get default base URL for the provider
     */
    protected abstract String getDefaultBaseUrl();

    /**
     * Get environment variable name for API key
     */
    protected abstract String getApiKeyEnvironmentVariable();

    /**
     * Check if provider supports function calling
     */
    protected boolean supportsFunctionCalling() {
        return false;
    }

    /**
     * Check if provider supports multimodal inputs
     */
    protected boolean supportsMultimodal() {
        return false;
    }

    /**
     * Check if provider supports embeddings
     */
    protected boolean supportsEmbeddings() {
        return false;
    }

    /**
     * Get max context tokens (cloud provider specific)
     */
    protected int getMaxContextTokens() {
        return getConfigValue("max-context-tokens", 8192);
    }

    /**
     * Get max output tokens (cloud provider specific)
     */
    protected int getMaxOutputTokens() {
        return getConfigValue("max-output-tokens", 4096);
    }

    /**
     * Track request
     */
    protected void trackRequest() {
        totalRequests.incrementAndGet();
    }

    /**
     * Track error
     */
    protected void trackError() {
        totalErrors.incrementAndGet();
    }

    /**
     * Get API key (for internal use)
     */
    protected String getApiKey() {
        return apiKey;
    }

    /**
     * Get base URL
     */
    protected String getBaseUrl() {
        return baseUrl;
    }

    /**
     * Get request timeout
     */
    protected Duration getRequestTimeout() {
        return requestTimeout;
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/adapter/LocalProviderAdapter.java
Size: 4.1 KB | Modified: 2026-01-19 14:05:59
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.adapter;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.inference.api.TenantContext;
import tech.kayys.wayang.inference.providers.loader.ModelLoader;
import tech.kayys.wayang.inference.providers.session.SessionManager;
import tech.kayys.wayang.inference.providers.spi.ProviderCapabilities;
import tech.kayys.wayang.inference.providers.spi.ProviderHealth;

import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Base adapter for local model execution (GGUF, ONNX, etc.)
 * Handles model loading, session management, and resource cleanup.
 */
public abstract class LocalProviderAdapter extends AbstractProvider {

    protected ModelLoader modelLoader;
    protected SessionManager sessionManager;
    protected final Map<String, Path> loadedModels = new ConcurrentHashMap<>();

    @Override
    protected Uni<Void> doInitialize(Map<String, Object> config, TenantContext tenant) {
        return Uni.createFrom().item(() -> {
            this.modelLoader = createModelLoader(config);
            this.sessionManager = createSessionManager(config);
            
            log.infof("Local provider %s initialized with model loader and session manager",
                providerId());
            
            return null;
        });
    }

    @Override
    protected Uni<ProviderHealth> doHealthCheck() {
        return Uni.createFrom().item(() -> {
            if (modelLoader == null || sessionManager == null) {
                return ProviderHealth.unhealthy("Provider components not initialized");
            }

            long activeModels = loadedModels.size();
            int activeSessions = sessionManager.activeSessionCount();
            
            return ProviderHealth.builder()
                .status(ProviderHealth.Status.HEALTHY)
                .message("Local provider operational")
                .detail("loaded_models", activeModels)
                .detail("active_sessions", activeSessions)
                .build();
        });
    }

    @Override
    protected Uni<Void> doShutdown() {
        return Uni.createFrom().item(() -> {
            if (sessionManager != null) {
                sessionManager.shutdown();
            }
            
            loadedModels.clear();
            
            log.infof("Local provider %s shutdown complete", providerId());
            return null;
        });
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
            .streaming(supportsStreamingInternally())
            .functionCalling(false)
            .multimodal(false)
            .embeddings(false)
            .maxContextTokens(getMaxContextTokens())
            .maxOutputTokens(getMaxOutputTokens())
            .build();
    }

    /**
     * Load model if not already loaded
     */
    protected Uni<Path> ensureModelLoaded(String modelId) {
        Path cached = loadedModels.get(modelId);
        if (cached != null) {
            return Uni.createFrom().item(cached);
        }

        return modelLoader.load(modelId)
            .invoke(path -> {
                loadedModels.put(modelId, path);
                log.infof("Model %s loaded at %s", modelId, path);
            });
    }

    /**
     * Get model path
     */
    protected Optional<Path> getModelPath(String modelId) {
        return Optional.ofNullable(loadedModels.get(modelId));
    }

    /**
     * Create model loader instance
     */
    protected abstract ModelLoader createModelLoader(Map<String, Object> config);

    /**
     * Create session manager instance
     */
    protected abstract SessionManager createSessionManager(Map<String, Object> config);

    /**
     * Check if provider supports streaming internally
     */
    protected abstract boolean supportsStreamingInternally();

    /**
     * Get max context tokens supported
     */
    protected int getMaxContextTokens() {
        return getConfigValue("max-context-tokens", 4096);
    }

    /**
     * Get max output tokens supported
     */
    protected int getMaxOutputTokens() {
        return getConfigValue("max-output-tokens", 2048);
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/adapter/ProviderContext.java
Size: 2.5 KB | Modified: 2026-01-19 14:06:58
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.adapter;

import tech.kayys.wayang.inference.api.TenantContext;

import java.time.Instant;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Context object passed to providers during execution
 */
public final class ProviderContext {

    private final String requestId;
    private final TenantContext tenantContext;
    private final Instant startTime;
    private final Map<String, Object> attributes;

    public ProviderContext(String requestId, TenantContext tenantContext) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.tenantContext = tenantContext;
        this.startTime = Instant.now();
        this.attributes = new ConcurrentHashMap<>();
    }

    public String getRequestId() {
        return requestId;
    }

    public TenantContext getTenantContext() {
        return tenantContext;
    }

    public Instant getStartTime() {
        return startTime;
    }

    public long getElapsedMs() {
        return Instant.now().toEpochMilli() - startTime.toEpochMilli();
    }

    public void setAttribute(String key, Object value) {
        attributes.put(key, value);
    }

    @SuppressWarnings("unchecked")
    public <T> Optional<T> getAttribute(String key, Class<T> type) {
        Object value = attributes.get(key);
        if (value != null && type.isInstance(value)) {
            return Optional.of((T) value);
        }
        return Optional.empty();
    }

    public Map<String, Object> getAttributes() {
        return Map.copyOf(attributes);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId;
        private TenantContext tenantContext;

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder tenantContext(TenantContext tenantContext) {
            this.tenantContext = tenantContext;
            return this;
        }

        public ProviderContext build() {
            return new ProviderContext(requestId, tenantContext);
        }
    }

    @Override
    public String toString() {
        return "ProviderContext{" +
               "requestId='" + requestId + '\'' +
               ", tenant=" + (tenantContext != null ? tenantContext.getTenantId() : "null") +
               ", elapsed=" + getElapsedMs() + "ms" +
               '}';
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/adapter/StreamingAdapter.java
Size: 3.3 KB | Modified: 2026-01-19 14:06:40
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.adapter;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.inference.providers.spi.ProviderRequest;
import tech.kayys.wayang.inference.providers.spi.ProviderResponse;
import tech.kayys.wayang.inference.providers.spi.StreamingProvider;
import tech.kayys.wayang.inference.providers.streaming.ChunkProcessor;
import tech.kayys.wayang.inference.providers.streaming.StreamHandler;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Adapter that adds streaming capabilities to providers
 */
public abstract class StreamingAdapter extends AbstractProvider implements StreamingProvider {

    protected StreamHandler streamHandler;
    protected ChunkProcessor chunkProcessor;

    @Override
    public Multi<StreamChunk> stream(ProviderRequest request) {
        if (!isInitialized()) {
            return Multi.createFrom().failure(
                new IllegalStateException("Provider not initialized")
            );
        }

        String tenantId = request.getTenantContext() != null
            ? request.getTenantContext().getTenantId()
            : "default";

        return checkRateLimit(tenantId)
            .onItem().transformToMulti(v -> doStream(request))
            .onFailure().transform(this::handleFailure);
    }

    @Override
    protected Uni<ProviderResponse> doInfer(ProviderRequest request) {
        if (request.isStreaming()) {
            // Collect streaming chunks into single response
            return collectStreamedResponse(request);
        } else {
            return doNonStreamingInfer(request);
        }
    }

    /**
     * Provider-specific streaming implementation
     */
    protected abstract Multi<StreamChunk> doStream(ProviderRequest request);

    /**
     * Provider-specific non-streaming implementation
     */
    protected abstract Uni<ProviderResponse> doNonStreamingInfer(ProviderRequest request);

    /**
     * Collect streamed chunks into single response
     */
    protected Uni<ProviderResponse> collectStreamedResponse(ProviderRequest request) {
        StringBuilder content = new StringBuilder();
        AtomicInteger chunkCount = new AtomicInteger();

        long startTime = System.currentTimeMillis();

        return doStream(request)
            .onItem().invoke(chunk -> {
                if (!chunk.isLast()) {
                    content.append(chunk.delta());
                    chunkCount.incrementAndGet();
                }
            })
            .collect().last()
            .map(lastChunk -> {
                long duration = System.currentTimeMillis() - startTime;
                
                return ProviderResponse.builder()
                    .requestId(request.getRequestId())
                    .content(content.toString())
                    .model(request.getModel())
                    .finishReason(lastChunk.finishReason())
                    .durationMs(duration)
                    .metadata("chunks", chunkCount.get())
                    .metadata("streaming", true)
                    .build();
            });
    }

    /**
     * Create stream handler
     */
    protected abstract StreamHandler createStreamHandler();

    /**
     * Create chunk processor
     */
    protected abstract ChunkProcessor createChunkProcessor();
}
================================================================================

================================================================================
tech/kayys/golek/provider/circuit/CircuitBreaker.java
Size: 2.0 KB | Modified: 2026-01-19 14:18:16
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.circuit;

import java.time.Duration;
import java.util.concurrent.Callable;
import java.util.function.Supplier;

/**
 * Circuit breaker for fault tolerance.
 * 
 * Prevents cascading failures by failing fast when error threshold exceeded.
 * 
 * States:
 * - CLOSED: Normal operation, requests pass through
 * - OPEN: Error threshold exceeded, requests fail immediately
 * - HALF_OPEN: Testing if service recovered, limited requests allowed
 */
public interface CircuitBreaker {

    /**
     * Circuit states
     */
    enum State {
        CLOSED,
        OPEN,
        HALF_OPEN
    }

    /**
     * Execute callable with circuit breaker protection
     * 
     * @param callable operation to execute
     * @return result of operation
     * @throws CircuitBreakerOpenException if circuit is open
     * @throws Exception if operation fails
     */
    <T> T call(Callable<T> callable) throws Exception;

    /**
     * Execute supplier with circuit breaker protection
     * 
     * @param supplier operation to execute
     * @return result of operation
     * @throws CircuitBreakerOpenException if circuit is open
     */
    <T> T get(Supplier<T> supplier);

    /**
     * Execute runnable with circuit breaker protection
     * 
     * @param runnable operation to execute
     * @throws CircuitBreakerOpenException if circuit is open
     */
    void run(Runnable runnable);

    /**
     * Get current circuit state
     */
    State getState();

    /**
     * Get circuit metrics
     */
    CircuitMetrics getMetrics();

    /**
     * Manually trip circuit open
     */
    void tripOpen();

    /**
     * Manually reset circuit
     */
    void reset();

    /**
     * Circuit breaker metrics
     */
    record CircuitMetrics(
        State state,
        int failureCount,
        int successCount,
        int totalCalls,
        double failureRate,
        Duration timeSinceStateChange,
        boolean isCallPermitted
    ) {}
}
================================================================================

================================================================================
tech/kayys/golek/provider/circuit/CircuitBreakerOpenException.java
Size: 1000 B | Modified: 2026-01-19 14:18:35
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.circuit;

/**
 * Exception thrown when circuit breaker is open.
 */
public class CircuitBreakerOpenException extends RuntimeException {

    private final String circuitName;
    private final long estimatedRecoveryTimeMs;

    public CircuitBreakerOpenException(String circuitName) {
        this(circuitName, 0);
    }

    public CircuitBreakerOpenException(String circuitName, long estimatedRecoveryTimeMs) {
        super(String.format(
            "Circuit breaker '%s' is OPEN%s",
            circuitName,
            estimatedRecoveryTimeMs > 0 
                ? String.format(" (estimated recovery in %dms)", estimatedRecoveryTimeMs)
                : ""
        ));
        this.circuitName = circuitName;
        this.estimatedRecoveryTimeMs = estimatedRecoveryTimeMs;
    }

    public String getCircuitName() {
        return circuitName;
    }

    public long getEstimatedRecoveryTimeMs() {
        return estimatedRecoveryTimeMs;
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/circuit/DefaultCircuitBreaker.java
Size: 13.6 KB | Modified: 2026-01-19 14:18:51
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.circuit;

import org.jboss.logging.Logger;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * Default circuit breaker implementation with configurable thresholds.
 * 
 * Thread-safe and non-blocking for state queries.
 * Uses atomic operations for counters and CAS for state transitions.
 */
public class DefaultCircuitBreaker implements CircuitBreaker {

    private static final Logger LOG = Logger.getLogger(DefaultCircuitBreaker.class);

    private final String name;
    private final CircuitBreakerConfig config;
    private final AtomicReference<State> state;
    private final AtomicInteger failureCount;
    private final AtomicInteger successCount;
    private final AtomicLong stateChangeTime;
    private final Lock stateLock;
    private final Predicate<Throwable> failurePredicate;

    public DefaultCircuitBreaker(String name, CircuitBreakerConfig config) {
        this.name = name;
        this.config = config;
        this.state = new AtomicReference<>(State.CLOSED);
        this.failureCount = new AtomicInteger(0);
        this.successCount = new AtomicInteger(0);
        this.stateChangeTime = new AtomicLong(System.currentTimeMillis());
        this.stateLock = new ReentrantLock();
        this.failurePredicate = config.failurePredicate();

        LOG.infof("Created circuit breaker '%s' with config: %s", name, config);
    }

    @Override
    public <T> T call(Callable<T> callable) throws Exception {
        if (!permitCall()) {
            throw new CircuitBreakerOpenException(
                name,
                getEstimatedRecoveryTime()
            );
        }

        long startTime = System.nanoTime();
        try {
            T result = callable.call();
            onSuccess(System.nanoTime() - startTime);
            return result;

        } catch (Exception e) {
            onFailure(e, System.nanoTime() - startTime);
            throw e;
        }
    }

    @Override
    public <T> T get(Supplier<T> supplier) {
        try {
            return call(supplier::get);
        } catch (Exception e) {
            if (e instanceof RuntimeException) {
                throw (RuntimeException) e;
            }
            throw new RuntimeException("Circuit breaker execution failed", e);
        }
    }

    @Override
    public void run(Runnable runnable) {
        get(() -> {
            runnable.run();
            return null;
        });
    }

    @Override
    public State getState() {
        // Check if OPEN circuit should transition to HALF_OPEN
        if (state.get() == State.OPEN) {
            long timeSinceOpen = System.currentTimeMillis() - stateChangeTime.get();
            if (timeSinceOpen >= config.openDuration().toMillis()) {
                transitionToHalfOpen();
            }
        }
        return state.get();
    }

    @Override
    public CircuitMetrics getMetrics() {
        State currentState = getState();
        int failures = failureCount.get();
        int successes = successCount.get();
        int total = failures + successes;
        double failureRate = total > 0 ? (double) failures / total : 0.0;

        return new CircuitMetrics(
            currentState,
            failures,
            successes,
            total,
            failureRate,
            Duration.ofMillis(System.currentTimeMillis() - stateChangeTime.get()),
            permitCall()
        );
    }

    @Override
    public void tripOpen() {
        stateLock.lock();
        try {
            if (state.get() != State.OPEN) {
                LOG.warnf("Circuit breaker '%s' manually tripped OPEN", name);
                transitionTo(State.OPEN);
            }
        } finally {
            stateLock.unlock();
        }
    }

    @Override
    public void reset() {
        stateLock.lock();
        try {
            LOG.infof("Circuit breaker '%s' manually reset", name);
            failureCount.set(0);
            successCount.set(0);
            transitionTo(State.CLOSED);
        } finally {
            stateLock.unlock();
        }
    }

    /**
     * Check if call is permitted
     */
    private boolean permitCall() {
        State currentState = getState();

        return switch (currentState) {
            case CLOSED -> true;
            case OPEN -> false;
            case HALF_OPEN -> {
                // In HALF_OPEN, allow limited number of test calls
                int permitted = config.halfOpenPermits();
                int total = successCount.get() + failureCount.get();
                yield total < permitted;
            }
        };
    }

    /**
     * Handle successful call
     */
    private void onSuccess(long durationNanos) {
        successCount.incrementAndGet();

        State currentState = state.get();
        if (currentState == State.HALF_OPEN) {
            // Check if enough successful calls to close circuit
            if (successCount.get() >= config.halfOpenSuccessThreshold()) {
                transitionToClosed();
            }
        }

        LOG.tracef("Circuit '%s' success: duration=%.2fms, state=%s, successes=%d",
                  name, durationNanos / 1e6, currentState, successCount.get());
    }

    /**
     * Handle failed call
     */
    private void onFailure(Throwable throwable, long durationNanos) {
        // Check if this failure type should count
        if (!failurePredicate.test(throwable)) {
            LOG.tracef("Circuit '%s' ignoring failure type: %s",
                      name, throwable.getClass().getSimpleName());
            return;
        }

        int failures = failureCount.incrementAndGet();
        State currentState = state.get();

        LOG.debugf("Circuit '%s' failure: duration=%.2fms, state=%s, failures=%d, error=%s",
                  name, durationNanos / 1e6, currentState, failures,
                  throwable.getClass().getSimpleName());

        if (currentState == State.HALF_OPEN) {
            // Any failure in HALF_OPEN reopens circuit
            transitionToOpen();
        } else if (currentState == State.CLOSED) {
            // Check if failure threshold exceeded
            int total = successCount.get() + failures;
            if (total >= config.slidingWindowSize()) {
                double failureRate = (double) failures / total;
                if (failureRate >= config.failureRateThreshold()) {
                    transitionToOpen();
                }
            } else if (failures >= config.failureThreshold()) {
                // Absolute threshold check
                transitionToOpen();
            }
        }
    }

    /**
     * Transition to CLOSED state
     */
    private void transitionToClosed() {
        stateLock.lock();
        try {
            if (state.get() != State.CLOSED) {
                LOG.infof("Circuit breaker '%s' transitioning HALF_OPEN -> CLOSED", name);
                failureCount.set(0);
                successCount.set(0);
                transitionTo(State.CLOSED);
            }
        } finally {
            stateLock.unlock();
        }
    }

    /**
     * Transition to OPEN state
     */
    private void transitionToOpen() {
        stateLock.lock();
        try {
            if (state.get() != State.OPEN) {
                LOG.warnf("Circuit breaker '%s' transitioning %s -> OPEN (failures=%d, rate=%.2f%%)",
                         name, state.get(), failureCount.get(),
                         getMetrics().failureRate() * 100);
                transitionTo(State.OPEN);
            }
        } finally {
            stateLock.unlock();
        }
    }

    /**
     * Transition to HALF_OPEN state
     */
    private void transitionToHalfOpen() {
        stateLock.lock();
        try {
            if (state.compareAndSet(State.OPEN, State.HALF_OPEN)) {
                LOG.infof("Circuit breaker '%s' transitioning OPEN -> HALF_OPEN (testing recovery)",
                         name);
                failureCount.set(0);
                successCount.set(0);
                stateChangeTime.set(System.currentTimeMillis());
            }
        } finally {
            stateLock.unlock();
        }
    }

    /**
     * Transition to new state
     */
    private void transitionTo(State newState) {
        state.set(newState);
        stateChangeTime.set(System.currentTimeMillis());
    }

    /**
     * Get estimated recovery time in milliseconds
     */
    private long getEstimatedRecoveryTime() {
        if (state.get() != State.OPEN) {
            return 0;
        }

        long timeSinceOpen = System.currentTimeMillis() - stateChangeTime.get();
        long openDurationMs = config.openDuration().toMillis();

        return Math.max(0, openDurationMs - timeSinceOpen);
    }

    @Override
    public String toString() {
        CircuitMetrics metrics = getMetrics();
        return String.format(
            "CircuitBreaker{name='%s', state=%s, failures=%d, successes=%d, rate=%.2f%%}",
            name, metrics.state(), metrics.failureCount(), metrics.successCount(),
            metrics.failureRate() * 100
        );
    }

    /**
     * Circuit breaker configuration
     */
    public record CircuitBreakerConfig(
        int failureThreshold,
        double failureRateThreshold,
        int slidingWindowSize,
        Duration openDuration,
        int halfOpenPermits,
        int halfOpenSuccessThreshold,
        Predicate<Throwable> failurePredicate
    ) {
        public CircuitBreakerConfig {
            if (failureThreshold <= 0) {
                throw new IllegalArgumentException("failureThreshold must be positive");
            }
            if (failureRateThreshold <= 0 || failureRateThreshold > 1) {
                throw new IllegalArgumentException("failureRateThreshold must be in (0,1]");
            }
            if (slidingWindowSize < failureThreshold) {
                throw new IllegalArgumentException(
                    "slidingWindowSize must be >= failureThreshold");
            }
            if (openDuration == null || openDuration.isNegative()) {
                throw new IllegalArgumentException("openDuration must be positive");
            }
            if (halfOpenPermits <= 0) {
                throw new IllegalArgumentException("halfOpenPermits must be positive");
            }
            if (halfOpenSuccessThreshold <= 0 || 
                halfOpenSuccessThreshold > halfOpenPermits) {
                throw new IllegalArgumentException(
                    "halfOpenSuccessThreshold must be in (0, halfOpenPermits]");
            }
            if (failurePredicate == null) {
                throw new IllegalArgumentException("failurePredicate cannot be null");
            }
        }

        /**
         * Default configuration
         */
        public static CircuitBreakerConfig defaults() {
            return new CircuitBreakerConfig(
                5,                                    // 5 failures
                0.5,                                  // 50% failure rate
                10,                                   // 10 call window
                Duration.ofSeconds(60),               // 60s open duration
                3,                                    // 3 test calls in half-open
                2,                                    // 2 successes to close
                throwable -> true                     // Count all exceptions
            );
        }

        /**
         * Create builder
         */
        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private int failureThreshold = 5;
            private double failureRateThreshold = 0.5;
            private int slidingWindowSize = 10;
            private Duration openDuration = Duration.ofSeconds(60);
            private int halfOpenPermits = 3;
            private int halfOpenSuccessThreshold = 2;
            private Predicate<Throwable> failurePredicate = throwable -> true;

            public Builder failureThreshold(int threshold) {
                this.failureThreshold = threshold;
                return this;
            }

            public Builder failureRateThreshold(double threshold) {
                this.failureRateThreshold = threshold;
                return this;
            }

            public Builder slidingWindowSize(int size) {
                this.slidingWindowSize = size;
                return this;
            }

            public Builder openDuration(Duration duration) {
                this.openDuration = duration;
                return this;
            }

            public Builder halfOpenPermits(int permits) {
                this.halfOpenPermits = permits;
                return this;
            }

            public Builder halfOpenSuccessThreshold(int threshold) {
                this.halfOpenSuccessThreshold = threshold;
                return this;
            }

            public Builder failurePredicate(Predicate<Throwable> predicate) {
                this.failurePredicate = predicate;
                return this;
            }

            public CircuitBreakerConfig build() {
                return new CircuitBreakerConfig(
                    failureThreshold,
                    failureRateThreshold,
                    slidingWindowSize,
                    openDuration,
                    halfOpenPermits,
                    halfOpenSuccessThreshold,
                    failurePredicate
                );
            }
        }
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/loader/ArtifactResolver.java
Size: 616 B | Modified: 2026-01-19 14:07:44
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.loader;

import io.smallrye.mutiny.Uni;

import java.nio.file.Path;

/**
 * Resolves and downloads model artifacts
 */
public interface ArtifactResolver {

    /**
     * Resolve artifact location and download if needed
     */
    Uni<Path> resolve(String artifactId);

    /**
     * Check if artifact is available locally
     */
    boolean isAvailableLocally(String artifactId);

    /**
     * Get local path for artifact
     */
    Path getLocalPath(String artifactId);

    /**
     * Clear cached artifact
     */
    Uni<Void> clearCache(String artifactId);
}
================================================================================

================================================================================
tech/kayys/golek/provider/loader/CachedArtifactResolver.java
Size: 2.2 KB | Modified: 2026-01-19 14:09:09
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.loader;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Objects;

/**
 * Caching decorator for artifact resolvers
 */
public class CachedArtifactResolver implements ArtifactResolver {

    private static final Logger LOG = Logger.getLogger(CachedArtifactResolver.class);

    private final ArtifactResolver delegate;
    private final Cache<String, Path> cache;

    public CachedArtifactResolver(ArtifactResolver delegate, int maxSize, Duration ttl) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.cache = Caffeine.newBuilder()
            .maximumSize(maxSize)
            .expireAfterWrite(ttl)
            .recordStats()
            .build();
    }

    @Override
    public Uni<Path> resolve(String artifactId) {
        Path cached = cache.getIfPresent(artifactId);
        if (cached != null) {
            LOG.debugf("Cache hit for artifact: %s", artifactId);
            return Uni.createFrom().item(cached);
        }

        LOG.debugf("Cache miss for artifact: %s", artifactId);
        return delegate.resolve(artifactId)
            .invoke(path -> cache.put(artifactId, path));
    }

    @Override
    public boolean isAvailableLocally(String artifactId) {
        return cache.getIfPresent(artifactId) != null || 
               delegate.isAvailableLocally(artifactId);
    }

    @Override
    public Path getLocalPath(String artifactId) {
        Path cached = cache.getIfPresent(artifactId);
        return cached != null ? cached : delegate.getLocalPath(artifactId);
    }

    @Override
    public Uni<Void> clearCache(String artifactId) {
        return Uni.createFrom().item(() -> {
            cache.invalidate(artifactId);
            return null;
        }).chain(() -> delegate.clearCache(artifactId));
    }

    public void clearAll() {
        cache.invalidateAll();
        LOG.info("Cleared all cached artifacts");
    }

    public long cacheSize() {
        return cache.estimatedSize();
    }

    public double hitRate() {
        return cache.stats().hitRate();
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/loader/LocalArtifactResolver.java
Size: 2.7 KB | Modified: 2026-01-19 14:08:04
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.loader;

import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Resolves artifacts from local filesystem
 */
public class LocalArtifactResolver implements ArtifactResolver {

    private static final Logger LOG = Logger.getLogger(LocalArtifactResolver.class);

    private final Path basePath;
    private final Map<String, Path> cache = new ConcurrentHashMap<>();

    public LocalArtifactResolver(String basePath) {
        this.basePath = Paths.get(basePath);
        ensureDirectoryExists(this.basePath);
    }

    @Override
    public Uni<Path> resolve(String artifactId) {
        return Uni.createFrom().item(() -> {
            Path cached = cache.get(artifactId);
            if (cached != null && Files.exists(cached)) {
                LOG.debugf("Artifact %s found in cache: %s", artifactId, cached);
                return cached;
            }

            Path artifactPath = basePath.resolve(artifactId);
            
            if (!Files.exists(artifactPath)) {
                throw new RuntimeException(
                    "Artifact not found: " + artifactId + " at " + artifactPath
                );
            }

            cache.put(artifactId, artifactPath);
            LOG.infof("Artifact %s resolved to: %s", artifactId, artifactPath);
            
            return artifactPath;
        });
    }

    @Override
    public boolean isAvailableLocally(String artifactId) {
        if (cache.containsKey(artifactId)) {
            return true;
        }
        
        Path artifactPath = basePath.resolve(artifactId);
        return Files.exists(artifactPath);
    }

    @Override
    public Path getLocalPath(String artifactId) {
        Path cached = cache.get(artifactId);
        if (cached != null) {
            return cached;
        }
        return basePath.resolve(artifactId);
    }

    @Override
    public Uni<Void> clearCache(String artifactId) {
        return Uni.createFrom().item(() -> {
            cache.remove(artifactId);
            LOG.debugf("Cleared cache for artifact: %s", artifactId);
            return null;
        });
    }

    private void ensureDirectoryExists(Path path) {
        try {
            if (!Files.exists(path)) {
                Files.createDirectories(path);
                LOG.infof("Created artifact directory: %s", path);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to create artifact directory: " + path, e);
        }
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/loader/ModelLoader.java
Size: 620 B | Modified: 2026-01-19 14:07:26
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.loader;

import io.smallrye.mutiny.Uni;

import java.nio.file.Path;

/**
 * Interface for loading model artifacts
 */
public interface ModelLoader {

    /**
     * Load model by ID
     * @param modelId Model identifier
     * @return Path to loaded model
     */
    Uni<Path> load(String modelId);

    /**
     * Check if model is already loaded
     */
    boolean isLoaded(String modelId);

    /**
     * Unload model and free resources
     */
    Uni<Void> unload(String modelId);

    /**
     * Get model path if loaded
     */
    Path getPath(String modelId);
}
================================================================================

================================================================================
tech/kayys/golek/provider/loader/RemoteArtifactResolver.java
Size: 5.0 KB | Modified: 2026-01-19 14:08:50
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.loader;

import io.smallrye.mutiny.Uni;
import io.vertx.core.buffer.Buffer;
import io.vertx.mutiny.core.Vertx;
import io.vertx.mutiny.ext.web.client.WebClient;
import io.vertx.ext.web.client.WebClientOptions;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Resolves artifacts from remote HTTP/HTTPS sources
 */
public class RemoteArtifactResolver implements ArtifactResolver {

private static final Logger LOG = Logger.getLogger(RemoteArtifactResolver.class);

    private final String baseUrl;
    private final Path cacheDir;
    private final WebClient webClient;
    private final Map<String, Path> cache = new ConcurrentHashMap<>();

    public RemoteArtifactResolver(Vertx vertx, String baseUrl, String cacheDir) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
        this.cacheDir = Paths.get(cacheDir);
        
        WebClientOptions options = new WebClientOptions()
            .setFollowRedirects(true)
            .setConnectTimeout(30000)
            .setIdleTimeout(60);
        
        this.webClient = WebClient.create(vertx, options);
        
        ensureDirectoryExists(this.cacheDir);
    }

    @Override
    public Uni<Path> resolve(String artifactId) {
        Path cached = cache.get(artifactId);
        if (cached != null && Files.exists(cached)) {
            LOG.debugf("Artifact %s found in cache: %s", artifactId, cached);
            return Uni.createFrom().item(cached);
        }

        Path localPath = cacheDir.resolve(artifactId);
        if (Files.exists(localPath)) {
            cache.put(artifactId, localPath);
            LOG.debugf("Artifact %s found at: %s", artifactId, localPath);
            return Uni.createFrom().item(localPath);
        }

        String url = baseUrl + artifactId;
        LOG.infof("Downloading artifact %s from: %s", artifactId, url);

        return downloadArtifact(url, localPath, artifactId);
    }

    @Override
    public boolean isAvailableLocally(String artifactId) {
        if (cache.containsKey(artifactId)) {
            return true;
        }
        
        Path localPath = cacheDir.resolve(artifactId);
        return Files.exists(localPath);
    }

    @Override
    public Path getLocalPath(String artifactId) {
        Path cached = cache.get(artifactId);
        if (cached != null) {
            return cached;
        }
        return cacheDir.resolve(artifactId);
    }

    @Override
    public Uni<Void> clearCache(String artifactId) {
        return Uni.createFrom().item(() -> {
            cache.remove(artifactId);
            
            Path localPath = cacheDir.resolve(artifactId);
            try {
                if (Files.exists(localPath)) {
                    Files.delete(localPath);
                    LOG.infof("Deleted cached artifact: %s", localPath);
                }
            } catch (IOException e) {
                LOG.warnf(e, "Failed to delete cached artifact: %s", localPath);
            }
            
            return null;
        });
    }

    private Uni<Path> downloadArtifact(String url, Path targetPath, String artifactId) {
        return webClient.getAbs(url)
            .send()
            .onItem().transformToUni(response -> {
                if (response.statusCode() == 200) {
                    return Uni.createFrom().item(() -> {
                        try {
                            Buffer body = response.body();
                            if (body == null || body.length() == 0) {
                                throw new RuntimeException("Empty response body for: " + url);
                            }
                            
                            Files.write(targetPath, body.getBytes());
                            cache.put(artifactId, targetPath);
                            
                            LOG.infof("Downloaded artifact %s (%d bytes) to: %s",
                                artifactId, body.length(), targetPath);
                            
                            return targetPath;
                        } catch (IOException e) {
                            throw new RuntimeException(
                                "Failed to write artifact to: " + targetPath, e
                            );
                        }
                    });
                } else {
                    return Uni.createFrom().failure(new RuntimeException(
                        "Failed to download artifact: HTTP " + response.statusCode()
                    ));
                }
            });
    }

    private void ensureDirectoryExists(Path path) {
        try {
            if (!Files.exists(path)) {
                Files.createDirectories(path);
                LOG.infof("Created cache directory: %s", path);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to create cache directory: " + path, e);
        }
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/ratelimit/RateLimiter.java
Size: 851 B | Modified: 2026-01-19 14:17:48
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.ratelimit;

/**
 * Rate limiter contract for controlling request throughput.
 * 
 * Implementations must be thread-safe.
 */
public interface RateLimiter {

    /**
     * Try to acquire a single permit.
     * 
     * @return true if permit acquired, false if rate limit exceeded
     */
    boolean tryAcquire();

    /**
     * Try to acquire multiple permits.
     * 
     * @param permits number of permits to acquire
     * @return true if all permits acquired, false otherwise
     */
    boolean tryAcquire(int permits);

    /**
     * Get number of available permits.
     * 
     * @return available permits (may be approximate)
     */
    int availablePermits();

    /**
     * Reset the rate limiter state.
     * Useful for testing or administrative operations.
     */
    void reset();
}
================================================================================

================================================================================
tech/kayys/golek/provider/ratelimit/SlidingWindowRateLimiter.java
Size: 6.9 KB | Modified: 2026-01-19 14:16:55
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.ratelimit;

import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Thread-safe sliding window rate limiter implementation.
 * 
 * Uses a deque to track timestamps of requests within the sliding window.
 * Automatically evicts expired timestamps to maintain accurate counts.
 * 
 * Performance characteristics:
 * - tryAcquire: O(n) worst case where n = expired timestamps
 * - availablePermits: O(n) worst case
 * - Memory: O(maxRequests) in worst case
 * 
 * Thread-safety: All operations are atomic and thread-safe.
 */
public class SlidingWindowRateLimiter implements RateLimiter {

    private static final Logger LOG = Logger.getLogger(SlidingWindowRateLimiter.class);

    private final int maxRequests;
    private final Duration window;
    private final ConcurrentLinkedDeque<Long> timestamps;
    private final AtomicInteger count;
    private final ReadWriteLock lock;
    
    // Metrics
    private final AtomicInteger rejectedCount;
    private final AtomicInteger acceptedCount;

    public SlidingWindowRateLimiter(int maxRequests, Duration window) {
        if (maxRequests <= 0) {
            throw new IllegalArgumentException("maxRequests must be positive");
        }
        if (window == null || window.isNegative() || window.isZero()) {
            throw new IllegalArgumentException("window must be positive duration");
        }
        
        this.maxRequests = maxRequests;
        this.window = window;
        this.timestamps = new ConcurrentLinkedDeque<>();
        this.count = new AtomicInteger(0);
        this.lock = new ReentrantReadWriteLock();
        this.rejectedCount = new AtomicInteger(0);
        this.acceptedCount = new AtomicInteger(0);
        
        LOG.infof("Created SlidingWindowRateLimiter: maxRequests=%d, window=%s", 
                  maxRequests, window);
    }

    @Override
    public boolean tryAcquire() {
        return tryAcquire(1);
    }

    @Override
    public boolean tryAcquire(int permits) {
        if (permits != 1) {
            throw new UnsupportedOperationException(
                "Sliding window only supports single permits, got: " + permits
            );
        }

        lock.writeLock().lock();
        try {
            long now = System.nanoTime();
            long windowStart = now - window.toNanos();

            // Remove old timestamps outside the window
            cleanupOldTimestamps(windowStart);

            // Check if we can accept new request
            if (count.get() < maxRequests) {
                timestamps.addLast(now);
                count.incrementAndGet();
                acceptedCount.incrementAndGet();
                
                LOG.tracef("Rate limit acquired: current=%d, max=%d", 
                          count.get(), maxRequests);
                return true;
            }

            rejectedCount.incrementAndGet();
            LOG.debugf("Rate limit exceeded: current=%d, max=%d", 
                      count.get(), maxRequests);
            return false;
            
        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public int availablePermits() {
        lock.readLock().lock();
        try {
            long now = System.nanoTime();
            long windowStart = now - window.toNanos();
            
            // Cleanup in read-only check
            cleanupOldTimestamps(windowStart);
            
            return Math.max(0, maxRequests - count.get());
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public void reset() {
        lock.writeLock().lock();
        try {
            timestamps.clear();
            count.set(0);
            rejectedCount.set(0);
            acceptedCount.set(0);
            
            LOG.infof("Rate limiter reset: maxRequests=%d, window=%s", 
                     maxRequests, window);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Remove timestamps that fall outside the current window.
     * Should be called while holding write lock.
     * 
     * @param windowStart Nanosecond timestamp of window start
     */
    private void cleanupOldTimestamps(long windowStart) {
        int removed = 0;
        
        // Remove from head while timestamps are outside window
        while (!timestamps.isEmpty()) {
            Long oldest = timestamps.peekFirst();
            if (oldest != null && oldest < windowStart) {
                timestamps.pollFirst();
                count.decrementAndGet();
                removed++;
            } else {
                break;
            }
        }
        
        if (removed > 0) {
            LOG.tracef("Cleaned up %d expired timestamps", removed);
        }
    }

    /**
     * Get metrics for monitoring
     */
    public RateLimiterMetrics getMetrics() {
        lock.readLock().lock();
        try {
            return new RateLimiterMetrics(
                maxRequests,
                count.get(),
                acceptedCount.get(),
                rejectedCount.get(),
                availablePermits()
            );
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Get time until next permit is available
     */
    public Duration getTimeUntilNextPermit() {
        lock.readLock().lock();
        try {
            if (count.get() < maxRequests) {
                return Duration.ZERO;
            }

            Long oldest = timestamps.peekFirst();
            if (oldest == null) {
                return Duration.ZERO;
            }

            long now = System.nanoTime();
            long oldestExpiry = oldest + window.toNanos();
            long nanosUntilAvailable = oldestExpiry - now;

            return nanosUntilAvailable > 0 
                ? Duration.ofNanos(nanosUntilAvailable)
                : Duration.ZERO;
                
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public String toString() {
        return String.format(
            "SlidingWindowRateLimiter{max=%d, window=%s, current=%d, available=%d}",
            maxRequests, window, count.get(), availablePermits()
        );
    }

    /**
     * Metrics snapshot for monitoring
     */
    public record RateLimiterMetrics(
        int maxRequests,
        int currentRequests,
        int totalAccepted,
        int totalRejected,
        int availablePermits
    ) {
        public double rejectionRate() {
            int total = totalAccepted + totalRejected;
            return total > 0 ? (double) totalRejected / total : 0.0;
        }

        public double utilizationRate() {
            return maxRequests > 0 ? (double) currentRequests / maxRequests : 0.0;
        }
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/ratelimit/TokenBucketRateLimiter.java
Size: 5.0 KB | Modified: 2026-01-19 14:17:29
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.ratelimit;

import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Token bucket rate limiter implementation.
 * 
 * Allows bursts up to capacity while maintaining steady refill rate.
 * More memory-efficient than sliding window for high-throughput scenarios.
 * 
 * Performance characteristics:
 * - tryAcquire: O(1)
 * - availablePermits: O(1)
 * - Memory: O(1)
 */
public class TokenBucketRateLimiter implements RateLimiter {

    private static final Logger LOG = Logger.getLogger(TokenBucketRateLimiter.class);

    private final int capacity;
    private final double refillRate; // tokens per nanosecond
    private final AtomicLong tokens; // stored as nanos * refillRate
    private final AtomicLong lastRefillTime;
    private final Lock lock;

    /**
     * Create token bucket rate limiter.
     * 
     * @param capacity maximum tokens (burst size)
     * @param refillPeriod time to fully refill bucket
     */
    public TokenBucketRateLimiter(int capacity, Duration refillPeriod) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be positive");
        }
        if (refillPeriod == null || refillPeriod.isNegative() || refillPeriod.isZero()) {
            throw new IllegalArgumentException("refillPeriod must be positive");
        }

        this.capacity = capacity;
        this.refillRate = (double) capacity / refillPeriod.toNanos();
        this.tokens = new AtomicLong((long) (capacity * 1e9)); // Store as nanos
        this.lastRefillTime = new AtomicLong(System.nanoTime());
        this.lock = new ReentrantLock();

        LOG.infof("Created TokenBucketRateLimiter: capacity=%d, refillPeriod=%s, rate=%.2e tokens/ns",
                  capacity, refillPeriod, refillRate);
    }

    @Override
    public boolean tryAcquire() {
        return tryAcquire(1);
    }

    @Override
    public boolean tryAcquire(int permits) {
        if (permits <= 0) {
            throw new IllegalArgumentException("permits must be positive");
        }
        if (permits > capacity) {
            return false; // Cannot acquire more than capacity
        }

        lock.lock();
        try {
            refillTokens();

            long required = (long) (permits * 1e9);
            long available = tokens.get();

            if (available >= required) {
                tokens.addAndGet(-required);
                LOG.tracef("Acquired %d permits, remaining: %.2f", 
                          permits, tokens.get() / 1e9);
                return true;
            }

            LOG.debugf("Insufficient tokens: required=%.2f, available=%.2f",
                      required / 1e9, available / 1e9);
            return false;

        } finally {
            lock.unlock();
        }
    }

    @Override
    public int availablePermits() {
        lock.lock();
        try {
            refillTokens();
            return (int) Math.min(capacity, tokens.get() / 1e9);
        } finally {
            lock.unlock();
        }
    }

    @Override
    public void reset() {
        lock.lock();
        try {
            tokens.set((long) (capacity * 1e9));
            lastRefillTime.set(System.nanoTime());
            LOG.infof("Token bucket reset: capacity=%d", capacity);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Refill tokens based on elapsed time.
     * Must be called while holding lock.
     */
    private void refillTokens() {
        long now = System.nanoTime();
        long last = lastRefillTime.get();
        long elapsed = now - last;

        if (elapsed > 0) {
            long toAdd = (long) (elapsed * refillRate * 1e9);
            if (toAdd > 0) {
                long newTokens = Math.min(
                    (long) (capacity * 1e9),
                    tokens.get() + toAdd
                );
                tokens.set(newTokens);
                lastRefillTime.set(now);

                LOG.tracef("Refilled tokens: added=%.2f, total=%.2f",
                          toAdd / 1e9, newTokens / 1e9);
            }
        }
    }

    /**
     * Get time until specified permits are available
     */
    public Duration getTimeUntilAvailable(int permits) {
        lock.lock();
        try {
            refillTokens();

            long required = (long) (permits * 1e9);
            long available = tokens.get();

            if (available >= required) {
                return Duration.ZERO;
            }

            long deficit = required - available;
            long nanosNeeded = (long) (deficit / refillRate);

            return Duration.ofNanos(nanosNeeded);

        } finally {
            lock.unlock();
        }
    }

    @Override
    public String toString() {
        return String.format(
            "TokenBucketRateLimiter{capacity=%d, available=%.2f, rate=%.2e/ns}",
            capacity, tokens.get() / 1e9, refillRate
        );
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/session/Session.java
Size: 4.1 KB | Modified: 2026-01-19 14:09:38
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.session;

import java.time.Instant;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Represents an active inference session.
 * Sessions manage resources and state for model execution.
 */
public final class Session implements AutoCloseable {

    private final String sessionId;
    private final String modelId;
    private final String tenantId;
    private final Instant createdAt;
    private final SessionConfig config;
    private final AtomicBoolean active;
    private final AtomicLong requestCount;
    private final AtomicLong lastAccessTime;

    private volatile Object nativeHandle; // Provider-specific handle

    public Session(String modelId, String tenantId, SessionConfig config) {
        this.sessionId = UUID.randomUUID().toString();
        this.modelId = Objects.requireNonNull(modelId, "modelId");
        this.tenantId = Objects.requireNonNull(tenantId, "tenantId");
        this.config = Objects.requireNonNull(config, "config");
        this.createdAt = Instant.now();
        this.active = new AtomicBoolean(true);
        this.requestCount = new AtomicLong(0);
        this.lastAccessTime = new AtomicLong(System.currentTimeMillis());
    }

    public String getSessionId() {
        return sessionId;
    }

    public String getModelId() {
        return modelId;
    }

    public String getTenantId() {
        return tenantId;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public SessionConfig getConfig() {
        return config;
    }

    public boolean isActive() {
        return active.get();
    }

    public long getRequestCount() {
        return requestCount.get();
    }

    public long getLastAccessTime() {
        return lastAccessTime.get();
    }

    public long getIdleTimeMs() {
        return System.currentTimeMillis() - lastAccessTime.get();
    }

    public long getAgeMs() {
        return System.currentTimeMillis() - createdAt.toEpochMilli();
    }

    /**
     * Mark session as accessed
     */
    public void recordAccess() {
        requestCount.incrementAndGet();
        lastAccessTime.set(System.currentTimeMillis());
    }

    /**
     * Set native handle (provider-specific)
     */
    public void setNativeHandle(Object handle) {
        this.nativeHandle = handle;
    }

    /**
     * Get native handle
     */
    @SuppressWarnings("unchecked")
    public <T> T getNativeHandle(Class<T> type) {
        if (nativeHandle != null && type.isInstance(nativeHandle)) {
            return (T) nativeHandle;
        }
        return null;
    }

    /**
     * Check if session has native handle
     */
    public boolean hasNativeHandle() {
        return nativeHandle != null;
    }

    /**
     * Check if session is idle
     */
    public boolean isIdle() {
        return getIdleTimeMs() > config.getMaxIdleTimeMs();
    }

    /**
     * Check if session is expired
     */
    public boolean isExpired() {
        return getAgeMs() > config.getMaxAgeMs();
    }

    /**
     * Check if session should be closed
     */
    public boolean shouldClose() {
        return !active.get() || isIdle() || isExpired();
    }

    @Override
    public void close() {
        if (active.compareAndSet(true, false)) {
            // Native handle cleanup should be done by session manager
            nativeHandle = null;
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Session session)) return false;
        return sessionId.equals(session.sessionId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(sessionId);
    }

    @Override
    public String toString() {
        return "Session{" +
               "id='" + sessionId + '\'' +
               ", model='" + modelId + '\'' +
               ", tenant='" + tenantId + '\'' +
               ", requests=" + requestCount.get() +
               ", idleMs=" + getIdleTimeMs() +
               ", active=" + active.get() +
               '}';
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/session/SessionConfig.java
Size: 2.8 KB | Modified: 2026-01-19 14:09:57
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.session;

import java.time.Duration;
import java.util.Objects;

/**
 * Configuration for session management
 */
public final class SessionConfig {

    private final int maxConcurrentSessions;
    private final Duration maxIdleTime;
    private final Duration maxAge;
    private final boolean reuseEnabled;
    private final int warmPoolSize;

    private SessionConfig(Builder builder) {
        this.maxConcurrentSessions = builder.maxConcurrentSessions;
        this.maxIdleTime = builder.maxIdleTime;
        this.maxAge = builder.maxAge;
        this.reuseEnabled = builder.reuseEnabled;
        this.warmPoolSize = builder.warmPoolSize;
    }

    public int getMaxConcurrentSessions() {
        return maxConcurrentSessions;
    }

    public Duration getMaxIdleTime() {
        return maxIdleTime;
    }

    public long getMaxIdleTimeMs() {
        return maxIdleTime.toMillis();
    }

    public Duration getMaxAge() {
        return maxAge;
    }

    public long getMaxAgeMs() {
        return maxAge.toMillis();
    }

    public boolean isReuseEnabled() {
        return reuseEnabled;
    }

    public int getWarmPoolSize() {
        return warmPoolSize;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static SessionConfig defaults() {
        return builder().build();
    }

    public static class Builder {
        private int maxConcurrentSessions = 10;
        private Duration maxIdleTime = Duration.ofMinutes(15);
        private Duration maxAge = Duration.ofHours(1);
        private boolean reuseEnabled = true;
        private int warmPoolSize = 2;

        public Builder maxConcurrentSessions(int maxConcurrentSessions) {
            this.maxConcurrentSessions = maxConcurrentSessions;
            return this;
        }

        public Builder maxIdleTime(Duration maxIdleTime) {
            this.maxIdleTime = maxIdleTime;
            return this;
        }

        public Builder maxAge(Duration maxAge) {
            this.maxAge = maxAge;
            return this;
        }

        public Builder reuseEnabled(boolean reuseEnabled) {
            this.reuseEnabled = reuseEnabled;
            return this;
        }

        public Builder warmPoolSize(int warmPoolSize) {
            this.warmPoolSize = warmPoolSize;
            return this;
        }

        public SessionConfig build() {
            return new SessionConfig(this);
        }
    }

    @Override
    public String toString() {
        return "SessionConfig{" +
               "maxConcurrent=" + maxConcurrentSessions +
               ", maxIdle=" + maxIdleTime +
               ", maxAge=" + maxAge +
               ", reuse=" + reuseEnabled +
               ", warmPool=" + warmPoolSize +
               '}';
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/session/SessionManager.java
Size: 4.1 KB | Modified: 2026-01-19 14:10:38
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.session;

import org.jboss.logging.Logger;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.*;

/**
 * Manages session pools for all models
 */
public class SessionManager {

    private static final Logger LOG = Logger.getLogger(SessionManager.class);

    private final SessionConfig config;
    private final Map<String, SessionPool> pools;
    private final ScheduledExecutorService cleanupExecutor;
    private volatile boolean shutdown;

    public SessionManager(SessionConfig config) {
        this.config = config;
        this.pools = new ConcurrentHashMap<>();
        this.cleanupExecutor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "session-cleanup");
            t.setDaemon(true);
            return t;
        });
        this.shutdown = false;

        // Schedule periodic cleanup
        cleanupExecutor.scheduleAtFixedRate(
            this::cleanupAllPools,
            1, 5, TimeUnit.MINUTES
        );
    }

    /**
     * Acquire session for model and tenant
     */
    public Optional<Session> acquireSession(String modelId, String tenantId) {
        return acquireSession(modelId, tenantId, 5000);
    }

    /**
     * Acquire session with timeout
     */
    public Optional<Session> acquireSession(
        String modelId, 
        String tenantId, 
        long timeoutMs
    ) {
        if (shutdown) {
            return Optional.empty();
        }

        String poolKey = getPoolKey(modelId, tenantId);
        SessionPool pool = pools.computeIfAbsent(poolKey, 
            k -> createPool(modelId, tenantId));

        return pool.acquire(timeoutMs);
    }

    /**
     * Release session back to pool
     */
    public void releaseSession(Session session) {
        if (session == null || shutdown) {
            return;
        }

        String poolKey = getPoolKey(session.getModelId(), session.getTenantId());
        SessionPool pool = pools.get(poolKey);
        
        if (pool != null) {
            pool.release(session);
        } else {
            LOG.warnf("No pool found for session %s, closing", session.getSessionId());
            session.close();
        }
    }

    /**
     * Get active session count
     */
    public int activeSessionCount() {
        return pools.values().stream()
            .mapToInt(pool -> pool.getStats().activeSessions())
            .sum();
    }

    /**
     * Get total session count
     */
    public int totalSessionCount() {
        return pools.values().stream()
            .mapToInt(pool -> pool.getStats().totalSessions())
            .sum();
    }

    /**
     * Get pool statistics for model
     */
    public Optional<SessionPool.PoolStats> getPoolStats(String modelId, String tenantId) {
        String poolKey = getPoolKey(modelId, tenantId);
        SessionPool pool = pools.get(poolKey);
        return pool != null ? Optional.of(pool.getStats()) : Optional.empty();
    }

    /**
     * Cleanup all pools
     */
    public void cleanupAllPools() {
        int totalCleaned = 0;
        for (SessionPool pool : pools.values()) {
            totalCleaned += pool.cleanup();
        }
        
        if (totalCleaned > 0) {
            LOG.infof("Cleaned %d idle/expired sessions across all pools", totalCleaned);
        }
    }

    /**
     * Shutdown manager and all pools
     */
    public void shutdown() {
        if (shutdown) {
            return;
        }

        shutdown = true;
        
        LOG.info("Shutting down session manager");

        cleanupExecutor.shutdownNow();

        for (SessionPool pool : pools.values()) {
            pool.shutdown();
        }

        pools.clear();
        
        LOG.info("Session manager shutdown complete");
    }

    private SessionPool createPool(String modelId, String tenantId) {
        LOG.infof("Creating session pool for model %s, tenant %s", modelId, tenantId);
        return new SessionPool(modelId, tenantId, config);
    }

    private String getPoolKey(String modelId, String tenantId) {
        return modelId + ":" + tenantId;
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/session/SessionPool.java
Size: 6.2 KB | Modified: 2026-01-19 14:10:18
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.session;

import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Pool of reusable sessions for a specific model
 */
public class SessionPool {

    private static final Logger LOG = Logger.getLogger(SessionPool.class);

    private final String modelId;
    private final String tenantId;
    private final SessionConfig config;
    private final BlockingQueue<Session> availableSessions;
    private final List<Session> allSessions;
    private final AtomicInteger activeCount;
    private volatile boolean shutdown;

    public SessionPool(String modelId, String tenantId, SessionConfig config) {
        this.modelId = modelId;
        this.tenantId = tenantId;
        this.config = config;
        this.availableSessions = new LinkedBlockingQueue<>();
        this.allSessions = new ArrayList<>();
        this.activeCount = new AtomicInteger(0);
        this.shutdown = false;
    }

    /**
     * Acquire a session from the pool
     */
    public Optional<Session> acquire(long timeoutMs) {
        if (shutdown) {
            return Optional.empty();
        }

        // Try to get from available pool first
        Session session = availableSessions.poll();
        if (session != null) {
            if (session.shouldClose()) {
                closeSession(session);
                return acquire(timeoutMs); // Retry
            }
            
            activeCount.incrementAndGet();
            session.recordAccess();
            LOG.debugf("Reused session %s for model %s", session.getSessionId(), modelId);
            return Optional.of(session);
        }

        // Create new session if under limit
        if (allSessions.size() < config.getMaxConcurrentSessions()) {
            Session newSession = createSession();
            if (newSession != null) {
                synchronized (allSessions) {
                    allSessions.add(newSession);
                }
                activeCount.incrementAndGet();
                LOG.infof("Created new session %s for model %s (total: %d)",
                    newSession.getSessionId(), modelId, allSessions.size());
                return Optional.of(newSession);
            }
        }

        // Wait for available session
        try {
            session = availableSessions.poll(timeoutMs, TimeUnit.MILLISECONDS);
            if (session != null) {
                if (session.shouldClose()) {
                    closeSession(session);
                    return Optional.empty();
                }
                activeCount.incrementAndGet();
                session.recordAccess();
                return Optional.of(session);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            LOG.warnf("Interrupted while waiting for session");
        }

        return Optional.empty();
    }

    /**
     * Release session back to pool
     */
    public void release(Session session) {
        if (session == null || shutdown) {
            return;
        }

        activeCount.decrementAndGet();

        if (session.shouldClose() || !config.isReuseEnabled()) {
            closeSession(session);
            return;
        }

        boolean offered = availableSessions.offer(session);
        if (!offered) {
            LOG.warnf("Failed to return session %s to pool, closing", session.getSessionId());
            closeSession(session);
        } else {
            LOG.debugf("Released session %s back to pool", session.getSessionId());
        }
    }

    /**
     * Cleanup idle and expired sessions
     */
    public int cleanup() {
        if (shutdown) {
            return 0;
        }

        int cleaned = 0;
        List<Session> toRemove = new ArrayList<>();

        // Check available sessions
        for (Session session : availableSessions) {
            if (session.shouldClose()) {
                toRemove.add(session);
            }
        }

        for (Session session : toRemove) {
            availableSessions.remove(session);
            closeSession(session);
            cleaned++;
        }

        LOG.debugf("Cleaned %d sessions from pool for model %s", cleaned, modelId);
        return cleaned;
    }

    /**
     * Get pool statistics
     */
    public PoolStats getStats() {
        return new PoolStats(
            allSessions.size(),
            activeCount.get(),
            availableSessions.size(),
            modelId,
            tenantId
        );
    }

    /**
     * Shutdown pool and close all sessions
     */
    public void shutdown() {
        if (shutdown) {
            return;
        }

        shutdown = true;
        
        LOG.infof("Shutting down session pool for model %s", modelId);

        synchronized (allSessions) {
            for (Session session : allSessions) {
                closeSession(session);
            }
            allSessions.clear();
        }

        availableSessions.clear();
        activeCount.set(0);
    }

    private Session createSession() {
        try {
            return new Session(modelId, tenantId, config);
        } catch (Exception e) {
            LOG.errorf(e, "Failed to create session for model %s", modelId);
            return null;
        }
    }

    private void closeSession(Session session) {
        try {
            session.close();
            synchronized (allSessions) {
                allSessions.remove(session);
            }
            LOG.debugf("Closed session %s", session.getSessionId());
        } catch (Exception e) {
            LOG.warnf(e, "Error closing session %s", session.getSessionId());
        }
    }

    public record PoolStats(
        int totalSessions,
        int activeSessions,
        int availableSessions,
        String modelId,
        String tenantId
    ) {
        @Override
        public String toString() {
            return String.format(
                "PoolStats{model=%s, total=%d, active=%d, available=%d}",
                modelId, totalSessions, activeSessions, availableSessions
            );
        }
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/spi/LLMProvider.java
Size: 1.4 KB | Modified: 2026-01-19 14:01:32
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.spi;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.inference.api.TenantContext;

import java.util.Map;

/**
 * Core SPI for all LLM providers.
 * Implementations must be thread-safe and support multi-tenancy.
 */
public interface LLMProvider {

    /**
     * Unique provider identifier (e.g., "ollama", "openai", "triton")
     */
    String providerId();

    /**
     * Provider metadata (version, capabilities, etc.)
     */
    ProviderMetadata metadata();

    /**
     * Initialize provider with configuration
     * Called once at startup per tenant
     */
    Uni<Void> initialize(Map<String, Object> config, TenantContext tenant);

    /**
     * Execute synchronous inference
     */
    Uni<ProviderResponse> infer(ProviderRequest request);

    /**
     * Check provider health
     */
    Uni<ProviderHealth> health();

    /**
     * Get provider capabilities
     */
    ProviderCapabilities capabilities();

    /**
     * Graceful shutdown
     */
    Uni<Void> shutdown();

    /**
     * Check if provider supports streaming
     */
    default boolean supportsStreaming() {
        return this instanceof StreamingProvider;
    }

    /**
     * Check if provider is available
     */
    default Uni<Boolean> isAvailable() {
        return health().map(h -> h.status() == ProviderHealth.Status.HEALTHY);
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/spi/ProviderCapabilities.java
Size: 5.8 KB | Modified: 2026-01-19 14:01:51
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.spi;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Immutable provider capabilities descriptor
 */
public final class ProviderCapabilities {

    private final boolean streaming;
    private final boolean functionCalling;
    private final boolean multimodal;
    private final boolean embeddings;
    private final int maxContextTokens;
    private final int maxOutputTokens;
    private final Set<String> supportedModels;
    private final List<String> supportedLanguages;
    private final Set<String> features;

    @JsonCreator
    public ProviderCapabilities(
        @JsonProperty("streaming") boolean streaming,
        @JsonProperty("functionCalling") boolean functionCalling,
        @JsonProperty("multimodal") boolean multimodal,
        @JsonProperty("embeddings") boolean embeddings,
        @JsonProperty("maxContextTokens") int maxContextTokens,
        @JsonProperty("maxOutputTokens") int maxOutputTokens,
        @JsonProperty("supportedModels") Set<String> supportedModels,
        @JsonProperty("supportedLanguages") List<String> supportedLanguages,
        @JsonProperty("features") Set<String> features
    ) {
        this.streaming = streaming;
        this.functionCalling = functionCalling;
        this.multimodal = multimodal;
        this.embeddings = embeddings;
        this.maxContextTokens = maxContextTokens;
        this.maxOutputTokens = maxOutputTokens;
        this.supportedModels = supportedModels != null 
            ? Set.copyOf(supportedModels) 
            : Collections.emptySet();
        this.supportedLanguages = supportedLanguages != null 
            ? List.copyOf(supportedLanguages) 
            : Collections.emptyList();
        this.features = features != null 
            ? Set.copyOf(features) 
            : Collections.emptySet();
    }

    // Getters
    public boolean isStreaming() { return streaming; }
    public boolean isFunctionCalling() { return functionCalling; }
    public boolean isMultimodal() { return multimodal; }
    public boolean isEmbeddings() { return embeddings; }
    public int getMaxContextTokens() { return maxContextTokens; }
    public int getMaxOutputTokens() { return maxOutputTokens; }
    public Set<String> getSupportedModels() { return supportedModels; }
    public List<String> getSupportedLanguages() { return supportedLanguages; }
    public Set<String> getFeatures() { return features; }

    public boolean supportsModel(String model) {
        return supportedModels.isEmpty() || supportedModels.contains(model);
    }

    public boolean hasFeature(String feature) {
        return features.contains(feature);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private boolean streaming = false;
        private boolean functionCalling = false;
        private boolean multimodal = false;
        private boolean embeddings = false;
        private int maxContextTokens = 4096;
        private int maxOutputTokens = 2048;
        private Set<String> supportedModels = Collections.emptySet();
        private List<String> supportedLanguages = List.of("en");
        private Set<String> features = Collections.emptySet();

        public Builder streaming(boolean streaming) {
            this.streaming = streaming;
            return this;
        }

        public Builder functionCalling(boolean functionCalling) {
            this.functionCalling = functionCalling;
            return this;
        }

        public Builder multimodal(boolean multimodal) {
            this.multimodal = multimodal;
            return this;
        }

        public Builder embeddings(boolean embeddings) {
            this.embeddings = embeddings;
            return this;
        }

        public Builder maxContextTokens(int maxContextTokens) {
            this.maxContextTokens = maxContextTokens;
            return this;
        }

        public Builder maxOutputTokens(int maxOutputTokens) {
            this.maxOutputTokens = maxOutputTokens;
            return this;
        }

        public Builder supportedModels(Set<String> supportedModels) {
            this.supportedModels = Set.copyOf(supportedModels);
            return this;
        }

        public Builder supportedLanguages(List<String> supportedLanguages) {
            this.supportedLanguages = List.copyOf(supportedLanguages);
            return this;
        }

        public Builder features(Set<String> features) {
            this.features = Set.copyOf(features);
            return this;
        }

        public ProviderCapabilities build() {
            return new ProviderCapabilities(
                streaming, functionCalling, multimodal, embeddings,
                maxContextTokens, maxOutputTokens, supportedModels,
                supportedLanguages, features
            );
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ProviderCapabilities that)) return false;
        return streaming == that.streaming &&
               functionCalling == that.functionCalling &&
               multimodal == that.multimodal &&
               maxContextTokens == that.maxContextTokens;
    }

    @Override
    public int hashCode() {
        return Objects.hash(streaming, functionCalling, multimodal, maxContextTokens);
    }

    @Override
    public String toString() {
        return "ProviderCapabilities{" +
               "streaming=" + streaming +
               ", functionCalling=" + functionCalling +
               ", multimodal=" + multimodal +
               ", maxContextTokens=" + maxContextTokens +
               ", features=" + features.size() +
               '}';
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/spi/ProviderException.java
Size: 1.4 KB | Modified: 2026-01-19 14:03:32
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.spi;

import tech.kayys.wayang.inference.api.ErrorPayload;

/**
 * Base exception for all provider errors
 */
public class ProviderException extends RuntimeException {

    private final String providerId;
    private final ErrorPayload errorPayload;
    private final boolean retryable;

    public ProviderException(String message) {
        this(null, message, null, false);
    }

    public ProviderException(String providerId, String message) {
        this(providerId, message, null, false);
    }

    public ProviderException(String providerId, String message, Throwable cause) {
        this(providerId, message, cause, false);
    }

    public ProviderException(
        String providerId, 
        String message, 
        Throwable cause, 
        boolean retryable
    ) {
        super(message, cause);
        this.providerId = providerId;
        this.retryable = retryable;
        this.errorPayload = ErrorPayload.builder()
            .type("ProviderError")
            .message(message)
            .retryable(retryable)
            .originNode(providerId)
            .detail("providerId", providerId)
            .build();
    }

    public String getProviderId() {
        return providerId;
    }

    public ErrorPayload getErrorPayload() {
        return errorPayload;
    }

    public boolean isRetryable() {
        return retryable;
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/spi/ProviderHealth.java
Size: 3.6 KB | Modified: 2026-01-19 14:03:14
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.spi;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Provider health status
 */
public final class ProviderHealth {

    public enum Status {
        HEALTHY,
        DEGRADED,
        UNHEALTHY,
        UNKNOWN
    }

    private final Status status;
    private final String message;
    private final Instant timestamp;
    private final Map<String, Object> details;

    @JsonCreator
    public ProviderHealth(
        @JsonProperty("status") Status status,
        @JsonProperty("message") String message,
        @JsonProperty("timestamp") Instant timestamp,
        @JsonProperty("details") Map<String, Object> details
    ) {
        this.status = Objects.requireNonNull(status, "status");
        this.message = message;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.details = details != null
            ? Collections.unmodifiableMap(new HashMap<>(details))
            : Collections.emptyMap();
    }

    public Status status() { return status; }
    public String message() { return message; }
    public Instant timestamp() { return timestamp; }
    public Map<String, Object> details() { return details; }

    public boolean isHealthy() {
        return status == Status.HEALTHY;
    }

    public boolean isDegraded() {
        return status == Status.DEGRADED;
    }

    public boolean isUnhealthy() {
        return status == Status.UNHEALTHY;
    }

    public static ProviderHealth healthy() {
        return new ProviderHealth(Status.HEALTHY, "Provider is healthy", null, null);
    }

    public static ProviderHealth healthy(String message) {
        return new ProviderHealth(Status.HEALTHY, message, null, null);
    }

    public static ProviderHealth degraded(String message) {
        return new ProviderHealth(Status.DEGRADED, message, null, null);
    }

    public static ProviderHealth unhealthy(String message) {
        return new ProviderHealth(Status.UNHEALTHY, message, null, null);
    }

    public static ProviderHealth unknown() {
        return new ProviderHealth(Status.UNKNOWN, "Health status unknown", null, null);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Status status = Status.UNKNOWN;
        private String message;
        private Instant timestamp = Instant.now();
        private final Map<String, Object> details = new HashMap<>();

        public Builder status(Status status) {
            this.status = status;
            return this;
        }

        public Builder message(String message) {
            this.message = message;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder detail(String key, Object value) {
            this.details.put(key, value);
            return this;
        }

        public Builder details(Map<String, Object> details) {
            this.details.putAll(details);
            return this;
        }

        public ProviderHealth build() {
            return new ProviderHealth(status, message, timestamp, details);
        }
    }
    @Override
    public String toString() {
        return "ProviderHealth{" +
               "status=" + status +
               ", message='" + message + '\'' +
               ", timestamp=" + timestamp +
               '}';
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/spi/ProviderMetadata.java
Size: 2.9 KB | Modified: 2026-01-19 14:02:50
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.spi;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Objects;

/**
 * Provider metadata and identification
 */
public final class ProviderMetadata {

    private final String providerId;
    private final String name;
    private final String version;
    private final String description;
    private final String vendor;
    private final String homepage;

    @JsonCreator
    public ProviderMetadata(
        @JsonProperty("providerId") String providerId,
        @JsonProperty("name") String name,
        @JsonProperty("version") String version,
        @JsonProperty("description") String description,
        @JsonProperty("vendor") String vendor,
        @JsonProperty("homepage") String homepage
    ) {
        this.providerId = Objects.requireNonNull(providerId, "providerId");
        this.name = Objects.requireNonNull(name, "name");
        this.version = Objects.requireNonNull(version, "version");
        this.description = description;
        this.vendor = vendor;
        this.homepage = homepage;
    }

    // Getters
    public String getProviderId() { return providerId; }
    public String getName() { return name; }
    public String getVersion() { return version; }
    public String getDescription() { return description; }
    public String getVendor() { return vendor; }
    public String getHomepage() { return homepage; }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String providerId;
        private String name;
        private String version;
        private String description;
        private String vendor;
        private String homepage;

        public Builder providerId(String providerId) {
            this.providerId = providerId;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder version(String version) {
            this.version = version;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder vendor(String vendor) {
            this.vendor = vendor;
            return this;
        }

        public Builder homepage(String homepage) {
            this.homepage = homepage;
            return this;
        }

        public ProviderMetadata build() {
            return new ProviderMetadata(
                providerId, name, version, description, vendor, homepage
            );
        }
    }

    @Override
    public String toString() {
        return "ProviderMetadata{" +
               "providerId='" + providerId + '\'' +
               ", name='" + name + '\'' +
               ", version='" + version + '\'' +
               '}';
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/spi/ProviderRegistry.java
Size: 2.2 KB | Modified: 2026-01-19 14:04:15
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.spi;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Central registry for all available providers.
 * Auto-discovers providers via CDI.
 */
@ApplicationScoped
public class ProviderRegistry {

    private static final Logger LOG = Logger.getLogger(ProviderRegistry.class);

    private final Map<String, LLMProvider> providers = new ConcurrentHashMap<>();

    @Inject
    Instance<LLMProvider> providerInstances;

    public void init() {
        providerInstances.forEach(provider -> {
            String id = provider.providerId();
            providers.put(id, provider);
            LOG.infof("Registered provider: %s (%s)", 
                id, provider.metadata().getName());
        });
    }

    /**
     * Get provider by ID
     */
    public Optional<LLMProvider> getProvider(String providerId) {
        return Optional.ofNullable(providers.get(providerId));
    }

    /**
     * Get all registered providers
     */
    public Collection<LLMProvider> getAllProviders() {
        return Collections.unmodifiableCollection(providers.values());
    }

    /**
     * Check if provider exists
     */
    public boolean hasProvider(String providerId) {
        return providers.containsKey(providerId);
    }

    /**
     * Get providers that support a specific model
     */
    public List<LLMProvider> getProvidersForModel(String model) {
        return providers.values().stream()
            .filter(p -> p.capabilities().supportsModel(model))
            .toList();
    }

    /**
     * Get streaming providers
     */
    public List<StreamingProvider> getStreamingProviders() {
        return providers.values().stream()
            .filter(p -> p instanceof StreamingProvider)
            .map(p -> (StreamingProvider) p)
            .toList();
    }

    /**
     * Get provider count
     */
    public int size() {
        return providers.size();
    }

    /**
     * Check if registry is empty
     */
    public boolean isEmpty() {
        return providers.isEmpty();
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/spi/ProviderRequest.java
Size: 5.4 KB | Modified: 2026-01-19 14:02:13
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.spi;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import tech.kayys.wayang.inference.api.Message;
import tech.kayys.wayang.inference.api.TenantContext;

import java.time.Duration;
import java.util.*;

/**
 * Normalized provider request.
 * Providers receive this standardized format regardless of original input.
 */
public final class ProviderRequest {

    @NotBlank
    private final String requestId;

    @NotBlank
    private final String model;

    @NotNull
    private final List<Message> messages;

    private final Map<String, Object> parameters;
    private final boolean streaming;
    private final Duration timeout;
    private final TenantContext tenantContext;

    @JsonCreator
    public ProviderRequest(
        @JsonProperty("requestId") String requestId,
        @JsonProperty("model") String model,
        @JsonProperty("messages") List<Message> messages,
        @JsonProperty("parameters") Map<String, Object> parameters,
        @JsonProperty("streaming") boolean streaming,
        @JsonProperty("timeout") Duration timeout,
        @JsonProperty("tenantContext") TenantContext tenantContext
    ) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.model = Objects.requireNonNull(model, "model");
        this.messages = Collections.unmodifiableList(new ArrayList<>(
            Objects.requireNonNull(messages, "messages")
        ));
        this.parameters = parameters != null 
            ? Collections.unmodifiableMap(new HashMap<>(parameters))
            : Collections.emptyMap();
        this.streaming = streaming;
        this.timeout = timeout != null ? timeout : Duration.ofSeconds(30);
        this.tenantContext = tenantContext;
    }

    // Getters
    public String getRequestId() { return requestId; }
    public String getModel() { return model; }
    public List<Message> getMessages() { return messages; }
    public Map<String, Object> getParameters() { return parameters; }
    public boolean isStreaming() { return streaming; }
    public Duration getTimeout() { return timeout; }
    public TenantContext getTenantContext() { return tenantContext; }

    // Parameter helpers
    public <T> Optional<T> getParameter(String key, Class<T> type) {
        Object value = parameters.get(key);
        if (value != null && type.isInstance(value)) {
            return Optional.of(type.cast(value));
        }
        return Optional.empty();
    }

    public double getTemperature() {
        return getParameter("temperature", Number.class)
            .map(Number::doubleValue)
            .orElse(0.7);
    }

    public int getMaxTokens() {
        return getParameter("max_tokens", Number.class)
            .map(Number::intValue)
            .orElse(2048);
    }

    public double getTopP() {
        return getParameter("top_p", Number.class)
            .map(Number::doubleValue)
            .orElse(1.0);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId = UUID.randomUUID().toString();
        private String model;
        private final List<Message> messages = new ArrayList<>();
        private final Map<String, Object> parameters = new HashMap<>();
        private boolean streaming = false;
        private Duration timeout = Duration.ofSeconds(30);
        private TenantContext tenantContext;

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder message(Message message) {
            this.messages.add(message);
            return this;
        }

        public Builder messages(List<Message> messages) {
            this.messages.addAll(messages);
            return this;
        }

        public Builder parameter(String key, Object value) {
            this.parameters.put(key, value);
            return this;
        }

        public Builder parameters(Map<String, Object> parameters) {
            this.parameters.putAll(parameters);
            return this;
        }

        public Builder streaming(boolean streaming) {
            this.streaming = streaming;
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public Builder tenantContext(TenantContext tenantContext) {
            this.tenantContext = tenantContext;
            return this;
        }

        public ProviderRequest build() {
            Objects.requireNonNull(model, "model is required");
            if (messages.isEmpty()) {
                throw new IllegalStateException("At least one message is required");
            }
            return new ProviderRequest(
                requestId, model, messages, parameters, streaming, timeout, tenantContext
            );
        }
    }

    @Override
    public String toString() {
        return "ProviderRequest{" +
               "requestId='" + requestId + '\'' +
               ", model='" + model + '\'' +
               ", messageCount=" + messages.size() +
               ", streaming=" + streaming +
               '}';
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/spi/ProviderResponse.java
Size: 5.2 KB | Modified: 2026-01-19 14:02:32
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.spi;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Normalized provider response
 */
public final class ProviderResponse {

    @NotBlank
    private final String requestId;

    @NotBlank
    private final String content;

    private final String model;
    private final String finishReason;
    private final int promptTokens;
    private final int completionTokens;
    private final int totalTokens;
    private final long durationMs;
    private final Instant timestamp;
    private final Map<String, Object> metadata;

    @JsonCreator
    public ProviderResponse(
        @JsonProperty("requestId") String requestId,
        @JsonProperty("content") String content,
        @JsonProperty("model") String model,
        @JsonProperty("finishReason") String finishReason,
        @JsonProperty("promptTokens") int promptTokens,
        @JsonProperty("completionTokens") int completionTokens,
        @JsonProperty("totalTokens") int totalTokens,
        @JsonProperty("durationMs") long durationMs,
        @JsonProperty("timestamp") Instant timestamp,
        @JsonProperty("metadata") Map<String, Object> metadata
    ) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.content = Objects.requireNonNull(content, "content");
        this.model = model;
        this.finishReason = finishReason;
        this.promptTokens = promptTokens;
        this.completionTokens = completionTokens;
        this.totalTokens = totalTokens;
        this.durationMs = durationMs;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.metadata = metadata != null
            ? Collections.unmodifiableMap(new HashMap<>(metadata))
            : Collections.emptyMap();
    }

    // Getters
    public String getRequestId() { return requestId; }
    public String getContent() { return content; }
    public String getModel() { return model; }
    public String getFinishReason() { return finishReason; }
    public int getPromptTokens() { return promptTokens; }
    public int getCompletionTokens() { return completionTokens; }
    public int getTotalTokens() { return totalTokens; }
    public long getDurationMs() { return durationMs; }
    public Instant getTimestamp() { return timestamp; }
    public Map<String, Object> getMetadata() { return metadata; }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId;
        private String content;
        private String model;
        private String finishReason = "stop";
        private int promptTokens;
        private int completionTokens;
        private int totalTokens;
        private long durationMs;
        private Instant timestamp = Instant.now();
        private final Map<String, Object> metadata = new HashMap<>();

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder content(String content) {
            this.content = content;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder finishReason(String finishReason) {
            this.finishReason = finishReason;
            return this;
        }

        public Builder promptTokens(int promptTokens) {
            this.promptTokens = promptTokens;
            return this;
        }

        public Builder completionTokens(int completionTokens) {
            this.completionTokens = completionTokens;
            return this;
        }

        public Builder totalTokens(int totalTokens) {
            this.totalTokens = totalTokens;
            return this;
        }

        public Builder durationMs(long durationMs) {
            this.durationMs = durationMs;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata.putAll(metadata);
            return this;
        }

        public ProviderResponse build() {
            Objects.requireNonNull(requestId, "requestId is required");
            Objects.requireNonNull(content, "content is required");
            return new ProviderResponse(
                requestId, content, model, finishReason,
                promptTokens, completionTokens, totalTokens,
                durationMs, timestamp, metadata
            );
        }
    }

    @Override
    public String toString() {
        return "ProviderResponse{" +
               "requestId='" + requestId + '\'' +
               ", model='" + model + '\'' +
               ", tokens=" + totalTokens +
               ", durationMs=" + durationMs +
               '}';
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/spi/StreamingProvider.java
Size: 932 B | Modified: 2026-01-19 14:03:57
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.spi;

import io.smallrye.mutiny.Multi;
import org.reactivestreams.Publisher;

/**
 * Extension interface for providers that support streaming
 */
public interface StreamingProvider extends LLMProvider {

    /**
     * Execute streaming inference
     * Returns a reactive stream of chunks
     */
    Multi<StreamChunk> stream(ProviderRequest request);

    /**
     * Stream chunk data
     */
    record StreamChunk(
        String requestId,
        String delta,
        boolean isLast,
        int index,
        String finishReason
    ) {
        public static StreamChunk of(String requestId, String delta, int index) {
            return new StreamChunk(requestId, delta, false, index, null);
        }

        public static StreamChunk last(String requestId, String finishReason) {
            return new StreamChunk(requestId, "", true, -1, finishReason);
        }
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/streaming/ChunkProcessor.java
Size: 562 B | Modified: 2026-01-19 14:11:23
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.streaming;

import tech.kayys.wayang.inference.providers.spi.StreamingProvider.StreamChunk;

/**
 * Processes raw chunks into StreamChunk objects
 */
public interface ChunkProcessor {

    /**
     * Process raw chunk data
     */
    StreamChunk process(String rawChunk, String requestId, int index);

    /**
     * Check if chunk indicates end of stream
     */
    boolean isEndOfStream(String rawChunk);

    /**
     * Extract finish reason from chunk
     */
    String extractFinishReason(String rawChunk);
}
================================================================================

================================================================================
tech/kayys/golek/provider/streaming/SSEStreamHandler.java
Size: 3.4 KB | Modified: 2026-01-19 14:11:44
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.streaming;

import io.smallrye.mutiny.Multi;
import io.vertx.core.http.HttpClientOptions;
import io.vertx.mutiny.core.Vertx;
import io.vertx.mutiny.core.buffer.Buffer;
import io.vertx.mutiny.core.http.HttpClient;
import io.vertx.mutiny.core.http.HttpClientRequest;
import io.vertx.mutiny.core.http.HttpClientResponse;
import org.jboss.logging.Logger;

import java.net.URI;
import java.time.Duration;

/**
 * Handles Server-Sent Events (SSE) streaming
 */
public class SSEStreamHandler implements StreamHandler {

    private static final Logger LOG = Logger.getLogger(SSEStreamHandler.class);

    private final Vertx vertx;
    private final HttpClient httpClient;
    private final Duration timeout;

    public SSEStreamHandler(Vertx vertx, Duration timeout) {
        this.vertx = vertx;
        this.timeout = timeout;
        
        HttpClientOptions options = new HttpClientOptions()
            .setConnectTimeout((int) timeout.toMillis())
            .setIdleTimeout((int) timeout.toSeconds())
            .setKeepAlive(true);
        
        this.httpClient = vertx.createHttpClient(options);
    }

    @Override
    public Multi<String> handleSSE(String url, String data) {
        try {
            URI uri = URI.create(url);
            int port = uri.getPort() > 0 ? uri.getPort() : 
                      ("https".equals(uri.getScheme()) ? 443 : 80);

            return httpClient.request(
                io.vertx.core.http.HttpMethod.POST,
                port,
                uri.getHost(),
                uri.getPath()
            )
            .onItem().transformToUni(request -> {
                request.putHeader("Accept", "text/event-stream");
                request.putHeader("Content-Type", "application/json");
                request.putHeader("Cache-Control", "no-cache");
                
                return request.send(Buffer.buffer(data));
            })
            .onItem().transformToMulti(this::processSSEResponse)
            .onFailure().invoke(ex -> 
                LOG.errorf(ex, "SSE streaming failed for URL: %s", url)
            );

        } catch (Exception e) {
            LOG.errorf(e, "Failed to initiate SSE stream to: %s", url);
            return Multi.createFrom().failure(e);
        }
    }

    @Override
    public Multi<String> handleWebSocket(String url, String data) {
        throw new UnsupportedOperationException(
            "SSEStreamHandler does not support WebSocket"
        );
    }

    @Override
    public void close() {
        if (httpClient != null) {
            httpClient.close();
        }
    }

    private Multi<String> processSSEResponse(HttpClientResponse response) {
        if (response.statusCode() != 200) {
            return Multi.createFrom().failure(new RuntimeException(
                "SSE stream failed with status: " + response.statusCode()
            ));
        }

        return response.toMulti()
            .onItem().transform(Buffer::toString)
            .onItem().transformToMultiAndConcatenate(this::parseSSEData)
            .select().where(line -> !line.isBlank());
    }

    private Multi<String> parseSSEData(String chunk) {
        // SSE format: "data: {json}\n\n"
        String[] lines = chunk.split("\n");
        
        return Multi.createFrom().items(lines)
            .filter(line -> line.startsWith("data: "))
            .map(line -> line.substring(6).trim())
            .filter(data -> !data.isEmpty() && !data.equals("[DONE]"));
    }
}
================================================================================

================================================================================
tech/kayys/golek/provider/streaming/StreamHandler.java
Size: 477 B | Modified: 2026-01-19 14:11:01
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.streaming;

import io.smallrye.mutiny.Multi;

/**
 * Handles streaming responses from providers
 */
public interface StreamHandler {

    /**
     * Handle server-sent events stream
     */
    Multi<String> handleSSE(String url, String data);

    /**
     * Handle WebSocket stream
     */
    Multi<String> handleWebSocket(String url, String data);

    /**
     * Close handler and cleanup resources
     */
    void close();
}
================================================================================

================================================================================
tech/kayys/golek/provider/streaming/WebSocketStreamHandler.java
Size: 3.1 KB | Modified: 2026-01-19 14:12:07
--------------------------------------------------------------------------------
package tech.kayys.wayang.inference.providers.streaming;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.operators.multi.processors.BroadcastProcessor;
import io.vertx.core.http.WebSocketConnectOptions;
import io.vertx.mutiny.core.Vertx;
import io.vertx.mutiny.core.http.HttpClient;
import io.vertx.mutiny.core.http.WebSocket;
import org.jboss.logging.Logger;

import java.net.URI;
import java.time.Duration;

/**
 * Handles WebSocket streaming
 */
public class WebSocketStreamHandler implements StreamHandler {

    private static final Logger LOG = Logger.getLogger(WebSocketStreamHandler.class);

    private final Vertx vertx;
    private final HttpClient httpClient;
    private final Duration timeout;
    private WebSocket activeWebSocket;

    public WebSocketStreamHandler(Vertx vertx, Duration timeout) {
        this.vertx = vertx;
        this.timeout = timeout;
        this.httpClient = vertx.createHttpClient();
    }

    @Override
    public Multi<String> handleSSE(String url, String data) {
        throw new UnsupportedOperationException(
            "WebSocketStreamHandler does not support SSE"
        );
    }

    @Override
    public Multi<String> handleWebSocket(String url, String data) {
        try {
            URI uri = URI.create(url);
            BroadcastProcessor<String> processor = BroadcastProcessor.create();

            WebSocketConnectOptions options = new WebSocketConnectOptions()
                .setHost(uri.getHost())
                .setPort(uri.getPort() > 0 ? uri.getPort() : 443)
                .setURI(uri.getPath())
                .setSsl("wss".equals(uri.getScheme()));

            httpClient.webSocket(options)
                .subscribe().with(
                    ws -> {
                        this.activeWebSocket = ws;
                        
                        ws.textMessageHandler(message -> {
                            processor.onNext(message);
                        });
                        
                        ws.closeHandler(v -> {
                            processor.onComplete();
                        });
                        
                        ws.exceptionHandler(ex -> {
                            LOG.errorf(ex, "WebSocket error");
                            processor.onError(ex);
                        });
                        
                        // Send initial data
                        ws.writeTextMessage(data);
                    },
                    error -> {
                        LOG.errorf(error, "Failed to connect WebSocket");
                        processor.onError(error);
                    }
                );

            return Multi.createFrom().publisher(processor);

        } catch (Exception e) {
            LOG.errorf(e, "Failed to initiate WebSocket stream to: %s", url);
            return Multi.createFrom().failure(e);
        }
    }

    @Override
    public void close() {
        if (activeWebSocket != null && !activeWebSocket.isClosed()) {
            activeWebSocket.close();
        }
        if (httpClient != null) {
            httpClient.close();
        }
    }
}
================================================================================


=== SUMMARY ===
Files processed: 33
Directories scanned: 12
Total input size: 118.3 KB
Output size: 129.6 KB
Processing time: 0.01 seconds
