package tech.kayys.gollek.engine.resilience;

import io.github.resilience4j.bulkhead.Bulkhead;
import io.github.resilience4j.bulkhead.BulkheadConfig;
import io.github.resilience4j.bulkhead.BulkheadRegistry;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.ratelimiter.RateLimiterConfig;
import io.github.resilience4j.ratelimiter.RateLimiterRegistry;
import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryConfig;
import io.github.resilience4j.retry.RetryRegistry;
import io.github.resilience4j.timelimiter.TimeLimiter;
import io.github.resilience4j.timelimiter.TimeLimiterConfig;
import io.github.resilience4j.timelimiter.TimeLimiterRegistry;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Resilience manager using Resilience4j for circuit breakers, bulkheads, retries, and rate limiters.
 *
 * <p>
 * Provides enterprise-grade reliability patterns:
 * <ul>
 * <li>Circuit Breaker - Prevent cascading failures</li>
 * <li>Bulkhead - Isolate resources (per-tenant isolation)</li>
 * <li>Retry - Automatic retry with exponential backoff</li>
 * <li>Rate Limiter - Control request throughput</li>
 * <li>Time Limiter - Timeout enforcement</li>
 * </ul>
 *
 * <h3>Configuration:</h3>
 * <ul>
 * <li>{@code gollek.resilience.circuit-breaker.*} - Circuit breaker settings</li>
 * <li>{@code gollek.resilience.bulkhead.*} - Bulkhead settings</li>
 * <li>{@code gollek.resilience.retry.*} - Retry settings</li>
 * <li>{@code gollek.resilience.rate-limiter.*} - Rate limiter settings</li>
 * <li>{@code gollek.resilience.time-limiter.*} - Time limiter settings</li>
 * </ul>
 *
 * @author Gollek Team
 * @version 1.0.0
 */
@ApplicationScoped
public class ResilienceManager {

    private static final Logger LOG = Logger.getLogger(ResilienceManager.class);

    // Registries
    private CircuitBreakerRegistry circuitBreakerRegistry;
    private BulkheadRegistry bulkheadRegistry;
    private RetryRegistry retryRegistry;
    private RateLimiterRegistry rateLimiterRegistry;
    private TimeLimiterRegistry timeLimiterRegistry;

    // Default configurations
    private CircuitBreakerConfig circuitBreakerConfig;
    private BulkheadConfig bulkheadConfig;
    private RetryConfig retryConfig;
    private RateLimiterConfig rateLimiterConfig;
    private TimeLimiterConfig timeLimiterConfig;

    // Per-tenant bulkheads
    private final Map<String, Bulkhead> tenantBulkheads = new ConcurrentHashMap<>();

    @PostConstruct
    synchronized void init() {
        LOG.info("Initializing ResilienceManager with Resilience4j");

        // Initialize Circuit Breaker
        circuitBreakerConfig = CircuitBreakerConfig.custom()
                .failureRateThreshold(50)
                .waitDurationInOpenState(Duration.ofSeconds(30))
                .slidingWindowSize(10)
                .minimumNumberOfCalls(5)
                .permittedNumberOfCallsInHalfOpenState(3)
                .automaticTransitionFromOpenToHalfOpenEnabled(true)
                .build();

        circuitBreakerRegistry = CircuitBreakerRegistry.of(circuitBreakerConfig);

        // Initialize Bulkhead
        bulkheadConfig = BulkheadConfig.custom()
                .maxConcurrentCalls(100)  // Max concurrent calls
                .maxWaitDuration(Duration.ofSeconds(10))  // Max wait time for permission
                .build();

        bulkheadRegistry = BulkheadRegistry.of(bulkheadConfig);

        // Initialize Retry
        retryConfig = RetryConfig.custom()
                .maxAttempts(3)
                .waitDuration(Duration.ofSeconds(1))
                .retryExceptions(Exception.class)
                .build();

        retryRegistry = RetryRegistry.of(retryConfig);

        // Initialize Rate Limiter
        rateLimiterConfig = RateLimiterConfig.custom()
                .limitRefreshPeriod(Duration.ofSeconds(1))  // Refresh limit every second
                .limitForPeriod(50)  // Allow 50 calls per period
                .timeoutDuration(Duration.ofSeconds(5))  // Wait up to 5s for permission
                .build();

        rateLimiterRegistry = RateLimiterRegistry.of(rateLimiterConfig);

        // Initialize Time Limiter
        timeLimiterConfig = TimeLimiterConfig.custom()
                .timeoutDuration(Duration.ofSeconds(30))  // 30 second timeout
                .cancelRunningFuture(true)  // Cancel on timeout
                .build();

        timeLimiterRegistry = TimeLimiterRegistry.of(timeLimiterConfig);

        LOG.info("ResilienceManager initialized with Resilience4j");
    }

    // ═══════════════════════════════════════════════════════════════
    // Circuit Breaker Methods
    // ═══════════════════════════════════════════════════════════════

    /**
     * Get or create a circuit breaker.
     *
     * @param name circuit breaker name
     * @return circuit breaker
     */
    public CircuitBreaker getCircuitBreaker(String name) {
        return circuitBreakerRegistry.circuitBreaker(name, circuitBreakerConfig);
    }

    /**
     * Get circuit breaker for a model.
     *
     * @param modelId model identifier
     * @return circuit breaker
     */
    public CircuitBreaker getModelCircuitBreaker(String modelId) {
        return getCircuitBreaker("model_" + modelId);
    }

    /**
     * Get circuit breaker for a provider.
     *
     * @param providerId provider identifier
     * @return circuit breaker
     */
    public CircuitBreaker getProviderCircuitBreaker(String providerId) {
        return getCircuitBreaker("provider_" + providerId);
    }

    /**
     * Get circuit breaker state.
     *
     * @param name circuit breaker name
     * @return state (CLOSED, OPEN, HALF_OPEN)
     */
    public String getCircuitBreakerState(String name) {
        CircuitBreaker cb = circuitBreakerRegistry.circuitBreaker(name);
        return cb != null ? cb.getState().name() : "UNKNOWN";
    }

    // ═══════════════════════════════════════════════════════════════
    // Bulkhead Methods
    // ═══════════════════════════════════════════════════════════════

    /**
     * Get or create a bulkhead.
     *
     * @param name bulkhead name
     * @return bulkhead
     */
    public Bulkhead getBulkhead(String name) {
        return bulkheadRegistry.bulkhead(name, bulkheadConfig);
    }

    /**
     * Get or create a per-tenant bulkhead.
     *
     * @param tenantId tenant identifier
     * @return bulkhead
     */
    public Bulkhead getTenantBulkhead(String tenantId) {
        return tenantBulkheads.computeIfAbsent(tenantId, id ->
            bulkheadRegistry.bulkhead("tenant_" + id, bulkheadConfig));
    }

    /**
     * Get bulkhead metrics.
     *
     * @param name bulkhead name
     * @return metrics map
     */
    public Map<String, Integer> getBulkheadMetrics(String name) {
        Bulkhead bulkhead = bulkheadRegistry.bulkhead(name);
        if (bulkhead == null) {
            return Map.of();
        }

        return Map.of(
                "available_calls", bulkhead.getMetrics().getAvailableConcurrentCalls(),
                "max_allowed", bulkhead.getMetrics().getMaxAllowedConcurrentCalls());
    }

    // ═══════════════════════════════════════════════════════════════
    // Retry Methods
    // ═══════════════════════════════════════════════════════════════

    /**
     * Get or create a retry.
     *
     * @param name retry name
     * @return retry
     */
    public Retry getRetry(String name) {
        return retryRegistry.retry(name, retryConfig);
    }

    /**
     * Get retry with custom config.
     *
     * @param name retry name
     * @param maxAttempts maximum retry attempts
     * @param waitDuration wait duration between retries
     * @return retry
     */
    public Retry getRetry(String name, int maxAttempts, Duration waitDuration) {
        RetryConfig customConfig = RetryConfig.custom()
                .maxAttempts(maxAttempts)
                .waitDuration(waitDuration)
                .retryExceptions(Exception.class)
                .build();

        return retryRegistry.retry(name, customConfig);
    }

    // ═══════════════════════════════════════════════════════════════
    // Rate Limiter Methods
    // ═══════════════════════════════════════════════════════════════

    /**
     * Get or create a rate limiter.
     *
     * @param name rate limiter name
     * @return rate limiter
     */
    public RateLimiter getRateLimiter(String name) {
        return rateLimiterRegistry.rateLimiter(name, rateLimiterConfig);
    }

    /**
     * Get rate limiter with custom config.
     *
     * @param name rate limiter name
     * @param limitForPeriod calls allowed per period
     * @param limitRefreshPeriod period duration
     * @return rate limiter
     */
    public RateLimiter getRateLimiter(String name, int limitForPeriod, Duration limitRefreshPeriod) {
        RateLimiterConfig customConfig = RateLimiterConfig.custom()
                .limitForPeriod(limitForPeriod)
                .limitRefreshPeriod(limitRefreshPeriod)
                .timeoutDuration(Duration.ofSeconds(5))
                .build();

        return rateLimiterRegistry.rateLimiter(name, customConfig);
    }

    // ═══════════════════════════════════════════════════════════════
    // Time Limiter Methods
    // ═══════════════════════════════════════════════════════════════

    /**
     * Get or create a time limiter.
     *
     * @param name time limiter name
     * @return time limiter
     */
    public TimeLimiter getTimeLimiter(String name) {
        return timeLimiterRegistry.timeLimiter(name, timeLimiterConfig);
    }

    /**
     * Get time limiter with custom timeout.
     *
     * @param name time limiter name
     * @param timeout timeout duration
     * @return time limiter
     */
    public TimeLimiter getTimeLimiter(String name, Duration timeout) {
        TimeLimiterConfig customConfig = TimeLimiterConfig.custom()
                .timeoutDuration(timeout)
                .cancelRunningFuture(true)
                .build();

        return timeLimiterRegistry.timeLimiter(name, customConfig);
    }

    // ═══════════════════════════════════════════════════════════════
    // Configuration Methods
    // ═══════════════════════════════════════════════════════════════

    /**
     * Update circuit breaker configuration.
     */
    public void updateCircuitBreakerConfig(float failureThreshold, Duration waitDuration, int slidingWindowSize) {
        this.circuitBreakerConfig = CircuitBreakerConfig.custom()
                .failureRateThreshold(failureThreshold)
                .waitDurationInOpenState(waitDuration)
                .slidingWindowSize(slidingWindowSize)
                .minimumNumberOfCalls(5)
                .permittedNumberOfCallsInHalfOpenState(3)
                .automaticTransitionFromOpenToHalfOpenEnabled(true)
                .build();

        LOG.infof("Updated circuit breaker config: failureThreshold=%.1f%%, waitDuration=%s, slidingWindowSize=%d",
                failureThreshold, waitDuration, slidingWindowSize);
    }

    /**
     * Update bulkhead configuration.
     */
    public void updateBulkheadConfig(int maxConcurrentCalls, Duration maxWaitDuration) {
        this.bulkheadConfig = BulkheadConfig.custom()
                .maxConcurrentCalls(maxConcurrentCalls)
                .maxWaitDuration(maxWaitDuration)
                .build();

        LOG.infof("Updated bulkhead config: maxConcurrentCalls=%d, maxWaitDuration=%s",
                maxConcurrentCalls, maxWaitDuration);
    }

    /**
     * Update retry configuration.
     */
    public void updateRetryConfig(int maxAttempts, Duration waitDuration) {
        this.retryConfig = RetryConfig.custom()
                .maxAttempts(maxAttempts)
                .waitDuration(waitDuration)
                .retryExceptions(Exception.class)
                .build();

        LOG.infof("Updated retry config: maxAttempts=%d, waitDuration=%s", maxAttempts, waitDuration);
    }

    // ═══════════════════════════════════════════════════════════════
    // Metrics and Monitoring
    // ═══════════════════════════════════════════════════════════════

    /**
     * Get all circuit breaker metrics.
     */
    public Map<String, Object> getAllCircuitBreakerMetrics() {
        Map<String, Object> result = new java.util.HashMap<>();
        circuitBreakerRegistry.getAllCircuitBreakers().forEach(cb -> {
            result.put(cb.getName(), Map.of(
                    "state", cb.getState().name(),
                    "failure_rate", cb.getMetrics().getFailureRate(),
                    "slow_call_rate", cb.getMetrics().getSlowCallRate(),
                    "total_calls", cb.getMetrics().getNumberOfBufferedCalls(),
                    "failed_calls", cb.getMetrics().getNumberOfFailedCalls()));
        });
        return result;
    }

    /**
     * Get all bulkhead metrics.
     */
    public Map<String, Object> getAllBulkheadMetrics() {
        Map<String, Object> result = new java.util.HashMap<>();
        bulkheadRegistry.getAllBulkheads().forEach(bh -> {
            result.put(bh.getName(), Map.of(
                    "available_calls", bh.getMetrics().getAvailableConcurrentCalls(),
                    "max_allowed", bh.getMetrics().getMaxAllowedConcurrentCalls()));
        });
        return result;
    }

    /**
     * Reset all registries (for testing).
     */
    public synchronized void reset() {
        circuitBreakerRegistry.getAllCircuitBreakers().forEach(CircuitBreaker::reset);
        tenantBulkheads.clear();
        LOG.info("ResilienceManager reset");
    }
}
