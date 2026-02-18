package tech.kayys.gollek.engine.reliability;

/**
 * Exception thrown when circuit breaker is open.
 */
public class CircuitBreakerOpenException extends RuntimeException {

    private final String circuitBreakerName;

    private final long estimatedRecoveryTimeMs;

    public String getCircuitBreakerName() {
        return circuitBreakerName;
    }

    public CircuitBreakerOpenException(String circuitBreakerName, long estimatedRecoveryTimeMs) {
        super(String.format(
                "Circuit breaker '%s' is OPEN%s",
                circuitBreakerName,
                estimatedRecoveryTimeMs > 0
                        ? String.format(" (estimated recovery in %dms)", estimatedRecoveryTimeMs)
                        : ""));
        this.circuitBreakerName = circuitBreakerName;
        this.estimatedRecoveryTimeMs = estimatedRecoveryTimeMs;
    }

    public long getEstimatedRecoveryTimeMs() {
        return estimatedRecoveryTimeMs;
    }
}