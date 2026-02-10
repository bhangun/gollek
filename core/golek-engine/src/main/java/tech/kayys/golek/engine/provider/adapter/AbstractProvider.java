package tech.kayys.golek.engine.provider.adapter;

import io.smallrye.mutiny.Uni;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import tech.kayys.golek.engine.reliability.CircuitBreaker;
import tech.kayys.golek.engine.reliability.DefaultCircuitBreaker;
import tech.kayys.golek.provider.core.ratelimit.RateLimiter;
import tech.kayys.golek.engine.ratelimit.TokenBucketRateLimiter;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.wayang.tenant.TenantId;
import tech.kayys.golek.spi.exception.ProviderException;
import tech.kayys.golek.spi.exception.ProviderException.ProviderInitializationException;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.provider.LLMProvider;
import tech.kayys.golek.spi.provider.ProviderConfig;
import tech.kayys.golek.spi.provider.ProviderHealth;
import tech.kayys.golek.spi.provider.ProviderRequest;

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

    // Store provider config to access metadata if needed
    protected ProviderConfig providerConfig;

    @jakarta.inject.Inject
    protected tech.kayys.golek.provider.core.quota.ProviderQuotaService quotaService;

    @ConfigProperty(name = "provider.health.cache.duration", defaultValue = "PT30S")
    protected Duration healthCacheDuration;

    @ConfigProperty(name = "provider.circuit-breaker.failure-threshold", defaultValue = "5")
    protected int circuitBreakerFailureThreshold;

    @ConfigProperty(name = "provider.circuit-breaker.timeout", defaultValue = "PT60S")
    protected Duration circuitBreakerTimeout;

    @ConfigProperty(name = "provider.rate-limit.enabled", defaultValue = "true")
    protected boolean rateLimitEnabled;

    @Override
    public final void initialize(ProviderConfig config) throws ProviderInitializationException {
        if (initialized.get()) {
            log.warnf("Provider %s already initialized", id());
            return;
        }

        log.infof("Initializing provider %s v%s", id(), version());

        this.providerConfig = config;
        this.configuration.putAll(config.getProperties());

        try {
            doInitialize(config.getProperties(), TenantContext.of(TenantId.of("system")))
                    .await().indefinitely(); // Blocking for now as initialize is synchronous interface
            initialized.set(true);
            log.infof("Provider %s initialized successfully", id());
        } catch (Exception e) {
            log.errorf(e, "Failed to initialize provider %s", id());
            throw new ProviderInitializationException("Failed to initialize provider " + id(), e);
        }
    }

    @Override
    public final Uni<InferenceResponse> infer(ProviderRequest request, TenantContext context) {
        if (!initialized.get()) {
            return Uni.createFrom().failure(
                    new ProviderException(id(), "Provider not initialized"));
        }

        String tenantId = context != null ? context.getTenantId().value() : "community";

        return checkQuota(context)
                .chain(() -> checkRateLimit(tenantId))
                .chain(() -> executeWithCircuitBreaker(request, context))
                .invoke(response -> quotaService.recordUsage(id(), response.getTokensUsed()))
                .onFailure().transform(this::handleFailure);
    }

    /**
     * Check if provider has quota
     */
    protected Uni<Void> checkQuota(TenantContext context) {
        return Uni.createFrom().item(() -> {
            if (!quotaService.hasQuota(id())) {
                String tenantId = context != null ? context.getTenantId().value() : "community";
                throw new tech.kayys.golek.spi.routing.QuotaExhaustedException(id(),
                        "Provider quota exhausted for: " + id() + ", tenant: " + tenantId);
            }
            return null;
        });
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
                    log.warnf(ex, "Health check failed for provider %s", id());
                    return ProviderHealth.unhealthy(ex.getMessage());
                });
    }

    @Override
    public final void shutdown() {
        if (!initialized.get()) {
            return;
        }

        log.infof("Shutting down provider %s", id());

        doShutdown();

        initialized.set(false);
        configuration.clear();
        rateLimiters.clear();
        circuitBreakers.clear();
        healthCache.set(null); // Clear health cache on shutdown

        log.infof("Provider %s shut down successfully", id());
    }

    /**
     * Provider-specific initialization logic
     */
    protected abstract Uni<Void> doInitialize(Map<String, Object> config, TenantContext tenant);

    /**
     * Provider-specific inference logic
     */
    protected abstract Uni<InferenceResponse> doInfer(ProviderRequest request);

    /**
     * Provider-specific health check
     */
    protected abstract Uni<ProviderHealth> doHealthCheck();

    /**
     * Provider-specific shutdown logic
     */
    protected void doShutdown() {
        // Default no-op
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
                id -> createRateLimiter(tenantId));

        return Uni.createFrom().item(() -> {
            if (!rateLimiter.tryAcquire()) {
                throw new ProviderException(
                        id(),
                        "Rate limit exceeded for tenant: " + tenantId,
                        null,
                        true); // Mark as retryable since rate limits may be temporary
            }
            return null;
        });
    }

    /**
     * Execute with circuit breaker protection
     */
    protected Uni<InferenceResponse> executeWithCircuitBreaker(
            ProviderRequest request,
            TenantContext context) {
        String tenantId = context != null ? context.getTenantId().value() : "community";

        CircuitBreaker circuitBreaker = circuitBreakers.computeIfAbsent(
                tenantId,
                id -> createCircuitBreaker(tenantId));

        return circuitBreaker.call(doInfer(request));
    }

    /**
     * Create rate limiter for tenant
     */
    protected RateLimiter createRateLimiter(String tenantId) {
        int requestsPerSecond = getConfigValue("rate-limit.requests-per-second", 10);
        int burst = getConfigValue("rate-limit.burst", 20);

        log.debugf("Creating rate limiter for tenant %s: %d req/s, burst %d",
                tenantId, requestsPerSecond, burst);

        // Ensure minimum values to prevent division by zero
        requestsPerSecond = Math.max(1, requestsPerSecond);
        burst = Math.max(1, burst);

        Duration refillPeriod = Duration.ofNanos(Math.max(1, (long) (1e9 * burst / requestsPerSecond)));
        return new TokenBucketRateLimiter(burst, refillPeriod);
    }

    /**
     * Create circuit breaker for tenant
     */
    protected CircuitBreaker createCircuitBreaker(String tenantId) {
        log.debugf("Creating circuit breaker for tenant %s: threshold %d, timeout %s",
                tenantId, circuitBreakerFailureThreshold, circuitBreakerTimeout);

        return new DefaultCircuitBreaker(
                "provider-" + tenantId,
                DefaultCircuitBreaker.CircuitBreakerConfig.builder()
                        .failureThreshold(circuitBreakerFailureThreshold)
                        .openDuration(circuitBreakerTimeout)
                        .build());
    }

    /**
     * Handle execution failures
     */
    protected Throwable handleFailure(Throwable ex) {
        if (ex instanceof ProviderException) {
            return ex;
        }

        log.errorf(ex, "Provider %s execution failed", id());

        // Check if the error is retryable
        boolean retryable = isRetryableError(ex);

        return new ProviderException(
                id(),
                "Provider execution failed: " + ex.getMessage(),
                ex,
                retryable);
    }

    /**
     * Determine if error is retryable
     */
    protected boolean isRetryableError(Throwable ex) {
        // Check the exception type first
        if (ex instanceof java.net.ConnectException ||
                ex instanceof java.net.SocketTimeoutException ||
                ex instanceof java.rmi.RemoteException) {
            return true;
        }

        // Check the exception message
        String message = ex.getMessage() != null ? ex.getMessage().toLowerCase() : "";
        return message.contains("timeout") ||
                message.contains("connection") ||
                message.contains("network") ||
                message.contains("unavailable") ||
                message.contains("refused") ||
                message.contains("reset") ||
                message.contains("broken pipe") ||
                message.contains("connection closed") ||
                message.contains("service unavailable") ||
                message.contains("gateway timeout") ||
                message.contains("too busy");
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

    /**
     * Get the provider ID
     */
    protected String getProviderId() {
        return id();
    }

    /**
     * Get the provider version
     */
    protected String getProviderVersion() {
        return version();
    }
}
