package tech.kayys.golek.provider.core.circuit;

import java.time.Duration;
import tech.kayys.golek.provider.core.circuit.CircuitBreaker.State;

/**
 * Circuit breaker metrics
 */
public record CircuitBreakerMetrics(
        State state,
        int failureCount,
        int successCount,
        int totalRequests,
        double failureRate,
        Duration timeSinceStateChange,
        boolean callsPermitted) {
}
