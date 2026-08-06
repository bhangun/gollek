package tech.kayys.gollek.observability;

import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.HealthCheckResponseBuilder;
import org.eclipse.microprofile.health.Liveness;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.InferenceEngine;

/**
 * Health check for the inference engine
 */
@Liveness
@ApplicationScoped
public class InferenceEngineHealthCheck implements HealthCheck {

    private static final Logger LOG = Logger.getLogger(InferenceEngineHealthCheck.class);

    @Inject
    Instance<InferenceEngine> engineInstance;

    @Override
    public HealthCheckResponse call() {
        HealthCheckResponseBuilder builder = HealthCheckResponse.named("inference-engine");

        try {
            if (engineInstance.isUnsatisfied()) {
                builder.up()
                        .withData("status", "NO_ENGINE")
                        .withData("reason", "No inference engine implementation available");
                return builder.build();
            }

            InferenceEngine engine = engineInstance.get();
            if (engine.isHealthy()) {
                var stats = engine.getStats();

                builder.up()
                        .withData("status", stats.status())
                        .withData("active_inferences", String.valueOf(stats.activeInferences()))
                        .withData("total_inferences", String.valueOf(stats.totalInferences()))
                        .withData("failed_inferences", String.valueOf(stats.failedInferences()))
                        .withData("avg_latency_ms", String.valueOf(stats.avgLatencyMs()));
            } else {
                builder.down()
                        .withData("status", "UNHEALTHY")
                        .withData("reason", "Engine is not initialized or unhealthy");
            }
        } catch (Exception e) {
            LOG.error("Health check failed", e);
            builder.down()
                    .withData("status", "ERROR")
                    .withData("error", e.getMessage());
        }

        return builder.build();
    }
}