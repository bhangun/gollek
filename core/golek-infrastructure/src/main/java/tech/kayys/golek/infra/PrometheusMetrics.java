package tech.kayys.golek.infra;

import io.micrometer.core.instrument.*;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.golek.api.observability.MetricsCollector;
import tech.kayys.wayang.tenant.TenantId;

import java.time.Duration;

/**
 * Centralized metrics collection using Micrometer.
 */
@ApplicationScoped
public class PrometheusMetrics implements MetricsCollector {

        @Inject
        MeterRegistry registry;

        public PrometheusMetrics() {
                // Required for CDI proxying
        }

        public PrometheusMetrics(MeterRegistry registry) {
                this.registry = registry;
        }

        @Override
        public void recordSuccess(
                        String provider,
                        String model,
                        TenantId tenant,
                        Duration duration) {

                Tags tags = Tags.of(
                                "status", "success",
                                "provider", provider,
                                "model", model,
                                "tenant", tenant.value());

                registry.counter("inference.requests.total", tags).increment();
                registry.timer("inference.duration", tags).record(duration);
        }

        @Override
        public void recordFailure(
                        String provider,
                        String model,
                        TenantId tenant,
                        String errorType) {

                Tags tags = Tags.of(
                                "status", "failure",
                                "provider", provider,
                                "model", model,
                                "tenant", tenant.value());

                registry.counter("inference.requests.total", tags).increment();

                Tags errorTags = Tags.of(
                                "type", errorType,
                                "provider", provider,
                                "tenant", tenant.value());
                registry.counter("inference.errors.total", errorTags).increment();
        }

        private int getActiveRequests() {
                // Implementation to track active requests if needed
                return 0;
        }
}