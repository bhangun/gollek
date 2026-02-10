package tech.kayys.wayang.memory.resilience;

import io.smallrye.faulttolerance.api.CircuitBreakerName;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.faulttolerance.CircuitBreaker;
import org.eclipse.microprofile.faulttolerance.Retry;
import org.eclipse.microprofile.faulttolerance.Timeout;

import java.time.temporal.ChronoUnit;

@ApplicationScoped
public class ResilientMemoryOperations {

    @Retry(maxRetries = 3, delay = 100, delayUnit = ChronoUnit.MILLIS)
    @Timeout(value = 5000, unit = ChronoUnit.MILLIS)
    @CircuitBreaker(
        requestVolumeThreshold = 10,
        failureRatio = 0.5,
        delay = 5000,
        delayUnit = ChronoUnit.MILLIS
    )
    @CircuitBreakerName("database-operations")
    public Uni<Void> resilientDatabaseOperation(Runnable operation) {
        return Uni.createFrom().item(() -> {
            operation.run();
            return null;
        });
    }

    @Retry(maxRetries = 2, delay = 50, delayUnit = ChronoUnit.MILLIS)
    @Timeout(value = 2000, unit = ChronoUnit.MILLIS)
    @CircuitBreaker(
        requestVolumeThreshold = 20,
        failureRatio = 0.3,
        delay = 3000,
        delayUnit = ChronoUnit.MILLIS
    )
    @CircuitBreakerName("cache-operations")
    public <T> Uni<T> resilientCacheOperation(java.util.concurrent.Callable<T> operation) {
        return Uni.createFrom().item(() -> {
            try {
                return operation.call();
            } catch (Exception e) {
                throw new RuntimeException("Cache operation failed", e);
            }
        });
    }
}