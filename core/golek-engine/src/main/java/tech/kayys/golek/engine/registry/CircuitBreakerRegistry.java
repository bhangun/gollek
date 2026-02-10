package tech.kayys.golek.engine.registry;

import java.util.Optional;
import tech.kayys.golek.engine.reliability.CircuitBreaker;

/**
 * Registry for CircuitBreakers.
 */
public interface CircuitBreakerRegistry {

    /**
     * Get a CircuitBreaker by ID.
     * 
     * @param id The ID of the circuit breaker.
     * @return Optional containing the CircuitBreaker if found.
     */
    Optional<CircuitBreaker> get(String id);

    /**
     * Reset the circuit breaker for the given ID.
     * 
     * @param id The ID of the circuit breaker to reset.
     */
    void reset(String id);
}
