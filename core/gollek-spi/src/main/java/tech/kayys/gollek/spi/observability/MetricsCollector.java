package tech.kayys.gollek.spi.observability;

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

        default void recordSuccessByApiKey(
                        String provider,
                        String model,
                        String apiKey,
                        Duration duration) {
                recordSuccess(provider, model, apiKey, duration);
        }

        void recordFailure(
                        String provider,
                        String model,
                        String tenant,
                        String errorType);

        default void recordFailureByApiKey(
                        String provider,
                        String model,
                        String apiKey,
                        String errorType) {
                recordFailure(provider, model, apiKey, errorType);
        }
}
