package tech.kayys.golek.engine.observability;

import java.time.Duration;
import tech.kayys.golek.engine.reliability.CircuitBreaker.State;

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
