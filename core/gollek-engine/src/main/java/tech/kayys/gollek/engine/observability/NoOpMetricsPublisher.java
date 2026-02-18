package tech.kayys.gollek.engine.observability;

import io.micrometer.core.instrument.Timer;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

/**
 * No-op implementation of MetricsPublisher for test environments.
 */
@ApplicationScoped
public class NoOpMetricsPublisher implements MetricsPublisher {

    private static final Logger LOG = Logger.getLogger(NoOpMetricsPublisher.class);

    @Override
    public void recordSuccess(String runnerName, String modelId, long duration) {
        LOG.tracef("Success: runner=%s, model=%s, duration=%dms", runnerName, modelId, duration);
    }

    @Override
    public void recordFailure(String runnerName, String modelId, String errorType) {
        LOG.tracef("Failure: runner=%s, model=%s, error=%s", runnerName, modelId, errorType);
    }

    @Override
    public Timer.Sample startTimer() {
        return Timer.start();
    }
}
