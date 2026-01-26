package tech.kayys.golek.infrastructure;

import io.micrometer.core.instrument.*;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

/**
 * Centralized metrics collection.
 */
@ApplicationScoped
public class PrometheusMetrics implements MetricsCollector {

    @Inject
    MeterRegistry registry;

    private final Counter inferenceCounter;
    private final Timer inferenceTimer;
    private final Counter errorCounter;
    private final Gauge activeRequests;

    public PrometheusMetrics(MeterRegistry registry) {
        this.registry = registry;

        this.inferenceCounter = Counter.builder("inference.requests.total")
                .description("Total inference requests")
                .tags("status", "provider", "model", "tenant")
                .register(registry);

        this.inferenceTimer = Timer.builder("inference.duration")
                .description("Inference request duration")
                .tags("provider", "model", "tenant")
                .register(registry);

        this.errorCounter = Counter.builder("inference.errors.total")
                .description("Total inference errors")
                .tags("type", "provider", "tenant")
                .register(registry);

        this.activeRequests = Gauge.builder("inference.requests.active",
                this::getActiveRequests)
                .description("Active inference requests")
                .register(registry);
    }

    @Override
    public void recordSuccess(
            String provider,
            String model,
            TenantId tenant,
            Duration duration) {
        inferenceCounter.increment(
                Tags.of(
                        "status", "success",
                        "provider", provider,
                        "model", model,
                        "tenant", tenant.value()));

        inferenceTimer.record(duration,
                Tags.of(
                        "provider", provider,
                        "model", model,
                        "tenant", tenant.value()));
    }

    @Override
    public void recordFailure(
            String provider,
            String model,
            TenantId tenant,
            String errorType) {
        inferenceCounter.increment(
                Tags.of(
                        "status", "failure",
                        "provider", provider,
                        "model", model,
                        "tenant", tenant.value()));

        errorCounter.increment(
                Tags.of(
                        "type", errorType,
                        "provider", provider,
                        "tenant", tenant.value()));
    }

    private int getActiveRequests() {
        // Implementation to track active requests
        return 0; // Placeholder
    }
}