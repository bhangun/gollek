package tech.kayys.golek.model.core;

import java.time.Duration;
import java.util.Optional;

/**
 * Interface for accessing runtime metrics for model runners.
 * This allows the model selection policy to make informed decisions
 * without depending on the engine module.
 */
public interface RunnerMetrics {

    /**
     * Get P95 latency for a specific runner and model
     */
    Optional<Duration> getP95Latency(String runnerName, String modelId);

    /**
     * Check if a runner is healthy
     */
    boolean isHealthy(String runnerName);

    /**
     * Get current load factor (0.0 to 1.0)
     */
    double getCurrentLoad(String runnerName);
}
