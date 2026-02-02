package tech.kayys.golek.api.observability;

import java.time.Duration;

/**
 * Interface for collecting metrics from inference operations.
 */
public interface MetricsCollector {

        void recordSuccess(
                        String provider,
                        String model,
                        String tenant,
                        Duration duration);

        void recordFailure(
                        String provider,
                        String model,
                        String tenant,
                        String errorType);
}
