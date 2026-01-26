package tech.kayys.golek.memory;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for managing circuit breakers per provider
 */
@ApplicationScoped
public class CircuitBreakerRegistry {

    private static final Logger LOG = Logger.getLogger(CircuitBreakerRegistry.class);

    @ConfigProperty(name = "circuit-breaker.failure.threshold", defaultValue = "5")
    int defaultFailureThreshold;

    @ConfigProperty(name = "circuit-breaker.failure.rate", defaultValue = "0.5")
    double defaultFailureRate;

    @ConfigProperty(name = "circuit-breaker.open.duration", defaultValue = "PT60S")
    Duration defaultOpenDuration;

    private final Map<String, CircuitBreaker> breakers = new ConcurrentHashMap<>();

    /**
     * Get or create circuit breaker for provider
     */
    public CircuitBreaker getOrCreate(String providerId) {
        return breakers.computeIfAbsent(providerId, id -> {
            LOG.infof("Creating circuit breaker for provider: %s", id);

            DefaultCircuitBreaker.CircuitBreakerConfig config = DefaultCircuitBreaker.CircuitBreakerConfig.builder()
                    .failureThreshold(defaultFailureThreshold)
                    .failureRateThreshold(defaultFailureRate)
                    .openDuration(defaultOpenDuration)
                    .minimumCalls(10)
                    .halfOpenPermits(3)
                    .halfOpenSuccessThreshold(2)
                    .build();

            return new DefaultCircuitBreaker(id, config);
        });
    }

    /**
     * Get circuit breaker if exists
     */
    public Optional<CircuitBreaker> get(String providerId) {
        return Optional.ofNullable(breakers.get(providerId));
    }

    /**
     * Get all circuit breakers
     */
    public Map<String, CircuitBreaker> getAll() {
        return Map.copyOf(breakers);
    }

    /**
     * Reset circuit breaker
     */
    public void reset(String providerId) {
        CircuitBreaker breaker = breakers.get(providerId);
        if (breaker != null) {
            breaker.reset();
            LOG.infof("Reset circuit breaker for: %s", providerId);
        }
    }

    /**
     * Reset all circuit breakers
     */
    public void resetAll() {
        breakers.values().forEach(CircuitBreaker::reset);
        LOG.info("Reset all circuit breakers");
    }

    /**
     * Remove circuit breaker
     */
    public void remove(String providerId) {
        CircuitBreaker removed = breakers.remove(providerId);
        if (removed != null) {
            LOG.infof("Removed circuit breaker for: %s", providerId);
        }
    }
}