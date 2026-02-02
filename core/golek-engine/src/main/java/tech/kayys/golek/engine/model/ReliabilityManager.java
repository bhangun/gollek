package tech.kayys.golek.engine.model;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.core.reliability.FallbackStrategy;
import tech.kayys.golek.core.reliability.FallbackStrategyRegistry;
import tech.kayys.golek.engine.reliability.CircuitBreaker;
import tech.kayys.golek.engine.reliability.DefaultCircuitBreaker;

import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Reliability manager providing circuit breakers, retries, and fallback
 * mechanisms
 */
public class ReliabilityManager {
    private static final Logger LOG = Logger.getLogger(ReliabilityManager.class);

    private final Map<String, CircuitBreaker> circuitBreakers = new ConcurrentHashMap<>();
    private final FallbackStrategyRegistry fallbackRegistry = new FallbackStrategyRegistry();

    /**
     * Execute an operation with circuit breaker protection
     */
    public <T> Uni<T> executeWithCircuitBreaker(String operationName, Supplier<Uni<T>> operation) {
        CircuitBreaker circuitBreaker = circuitBreakers.computeIfAbsent(
                operationName,
                name -> createDefaultCircuitBreaker(name));

        return Uni.createFrom().deferred(() -> {
            if (!circuitBreaker.getMetrics().isCallPermitted()) {
                return Uni.createFrom().failure(
                        new CircuitBreakerOpenException(operationName,
                                circuitBreaker.getMetrics().estimatedRecoveryTimeMs()));
            }

            return operation.get()
                    .onFailure().invoke(failure -> {
                        LOG.warnf("Operation %s failed: %s", operationName, failure.getMessage());
                        // The circuit breaker will automatically handle failure counting
                    });
        });
    }

    /**
     * Execute an operation with fallback strategy
     */
    public <T> Uni<T> executeWithFallback(String operationName,
            Supplier<Uni<T>> primaryOperation,
            Supplier<Uni<T>> fallbackOperation) {
        return executeWithCircuitBreaker(operationName, primaryOperation)
                .onFailure().recoverWithUni(failure -> {
                    LOG.warnf("Primary operation %s failed, executing fallback: %s",
                            operationName, failure.getMessage());

                    return fallbackOperation.get()
                            .onFailure().invoke(fallbackFailure -> {
                                LOG.errorf("Both primary and fallback operations failed for %s: %s",
                                        operationName, fallbackFailure.getMessage());
                            });
                });
    }

    /**
     * Execute an operation with retry logic
     */
    public <T> Uni<T> executeWithRetry(String operationName,
            Supplier<Uni<T>> operation,
            int maxRetries,
            Duration delay) {
        return executeWithCircuitBreaker(operationName, operation)
                .onFailure().retry()
                .withBackOff(delay)
                .atMost(maxRetries)
                .expireIn(Duration.ofMinutes(5)); // Expire retry attempts after 5 minutes
    }

    /**
     * Execute an operation with circuit breaker, retry, and fallback
     */
    public <T> Uni<T> executeResiliently(String operationName,
            Supplier<Uni<T>> primaryOperation,
            Supplier<Uni<T>> fallbackOperation,
            int maxRetries,
            Duration retryDelay) {
        Supplier<Uni<T>> operationWithRetry = () -> executeWithRetry(operationName, primaryOperation, maxRetries,
                retryDelay);

        return executeWithFallback(operationName, operationWithRetry, fallbackOperation);
    }

    /**
     * Register a fallback strategy
     */
    public void registerFallbackStrategy(String operationType, FallbackStrategy strategy) {
        fallbackRegistry.registerStrategy(operationType, strategy);
    }

    /**
     * Get a fallback strategy
     */
    public FallbackStrategy getFallbackStrategy(String operationType) {
        return fallbackRegistry.getStrategy(operationType);
    }

    /**
     * Create a default circuit breaker
     */
    private CircuitBreaker createDefaultCircuitBreaker(String name) {
        var config = DefaultCircuitBreaker.CircuitBreakerConfig.builder()
                .failureThreshold(5)
                .failureRateThreshold(0.5)
                .slidingWindowSize(10)
                .openDuration(Duration.ofSeconds(60))
                .halfOpenPermits(3)
                .halfOpenSuccessThreshold(2)
                .build();

        return new DefaultCircuitBreaker(name, config);
    }

    /**
     * Get circuit breaker by name
     */
    public CircuitBreaker getCircuitBreaker(String name) {
        return circuitBreakers.get(name);
    }

    /**
     * Reset a circuit breaker
     */
    public void resetCircuitBreaker(String name) {
        CircuitBreaker cb = circuitBreakers.get(name);
        if (cb != null) {
            cb.reset();
        }
    }

    /**
     * Trip a circuit breaker open
     */
    public void tripCircuitBreaker(String name) {
        CircuitBreaker cb = circuitBreakers.get(name);
        if (cb != null) {
            cb.tripOpen();
        }
    }

    /**
     * Get all circuit breaker metrics
     */
    public Map<String, CircuitBreaker.CircuitMetrics> getAllMetrics() {
        Map<String, CircuitBreaker.CircuitMetrics> metrics = new ConcurrentHashMap<>();
        circuitBreakers.forEach((name, cb) -> metrics.put(name, cb.getMetrics()));
        return metrics;
    }

    /**
     * Circuit breaker open exception
     */
    public static class CircuitBreakerOpenException extends RuntimeException {
        private final String operationName;
        private final long estimatedRecoveryTimeMs;

        public CircuitBreakerOpenException(String operationName, long estimatedRecoveryTimeMs) {
            super(String.format("Circuit breaker for operation '%s' is OPEN (recovery in ~%d ms)",
                    operationName, estimatedRecoveryTimeMs));
            this.operationName = operationName;
            this.estimatedRecoveryTimeMs = estimatedRecoveryTimeMs;
        }

        public String getOperationName() {
            return operationName;
        }

        public long getEstimatedRecoveryTimeMs() {
            return estimatedRecoveryTimeMs;
        }
    }
}