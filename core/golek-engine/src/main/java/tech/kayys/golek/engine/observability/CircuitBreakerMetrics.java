package tech.kayys.golek.engine.observability;

import java.time.Duration;
import tech.kayys.golek.engine.reliability.CircuitBreaker;
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
                boolean callsPermitted,
                long estimatedRecoveryTimeMs) implements CircuitBreaker.CircuitBreakerMetrics {

        @Override
        public State state() {
                return state;
        }

        @Override
        public boolean isCallPermitted() {
                return callsPermitted;
        }

        @Override
        public int getFailureCount() {
                return failureCount;
        }

        @Override
        public int getSuccessCount() {
                return successCount;
        }

        @Override
        public int getTotalRequests() {
                return totalRequests;
        }

        @Override
        public double failureRate() {
                return failureRate;
        }

        @Override
        public long estimatedRecoveryTimeMs() {
                return estimatedRecoveryTimeMs;
        }
}
