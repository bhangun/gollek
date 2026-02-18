package tech.kayys.gollek.engine.reliability;

import io.smallrye.mutiny.Uni;

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
        CLOSED, // Normal operation
        OPEN, // Circuit tripped, rejecting calls
        HALF_OPEN // Testing if service recovered
    }

    /**
     * Execute Uni with circuit breaker protection
     */
    <T> Uni<T> call(Uni<T> uni);

    /**
     * Manually trip circuit open
     */
    void tripOpen();

    /**
     * Reset circuit to closed
     */
    void reset();

    /**
     * Circuit breaker metrics
     */
    interface CircuitBreakerMetrics {
        State state();

        boolean isCallPermitted();

        int getFailureCount();

        int getSuccessCount();

        int getTotalRequests();

        double failureRate();

        long estimatedRecoveryTimeMs();
    }

    /**
     * Get circuit metrics
     */
    CircuitBreakerMetrics getMetrics();

    /**
     * Execute callable with circuit breaker protection
     * 
     * @param callable operation to execute
     * @return result of operation
     * @throws CircuitBreakerOpenException if circuit is open
     * @throws Exception                   if operation fails
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

}