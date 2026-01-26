package tech.kayys.golek.memory;

import io.smallrye.mutiny.Uni;

import java.time.Duration;
import java.util.concurrent.Callable;

/**
 * Circuit breaker pattern for provider fault tolerance.
 * States: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
 */
public interface CircuitBreaker {

    enum State {
        CLOSED, // Normal operation
        OPEN, // Failing, rejecting calls
        HALF_OPEN // Testing if recovered
    }

    /**
     * Execute callable with circuit breaker protection
     */
    <T> T call(Callable<T> callable) throws Exception;

    /**
     * Execute Uni with circuit breaker protection
     */
    <T> Uni<T> callAsync(Callable<Uni<T>> callable);

    /**
     * Get current state
     */
    State getState();

    /**
     * Force open the circuit
     */
    void tripOpen();

    /**
     * Force close the circuit
     */
    void reset();

    /**
     * Get circuit breaker metrics
     */
    CircuitBreakerMetrics getMetrics();

    /**
     * Circuit breaker metrics
     */
    interface CircuitBreakerMetrics {
        long successCount();

        long failureCount();

        long totalCalls();

        double failureRate();

        State currentState();

        long rejectedCalls();
    }
}