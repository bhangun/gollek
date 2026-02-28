package tech.kayys.gollek.engine.observability;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

@ApplicationScoped
public class AdapterRoutingMetricsCollector {

    private final MeterRegistry meterRegistry;
    private final String distributionMode;
    private final ConcurrentMap<String, Counter> filteredCounters = new ConcurrentHashMap<>();

    @Inject
    public AdapterRoutingMetricsCollector(
            MeterRegistry meterRegistry,
            @ConfigProperty(name = "gollek.distribution.mode", defaultValue = "standalone") String distributionMode) {
        this.meterRegistry = meterRegistry;
        this.distributionMode = distributionMode;
    }

    public AdapterRoutingMetricsCollector(MeterRegistry meterRegistry) {
        this(meterRegistry, "standalone");
    }

    public void recordProviderFiltered(
            String router,
            String providerId,
            String modelId,
            String tenantId,
            String reason) {
        String key = String.join("|", router, providerId, modelId, tenantId, reason, distributionMode);
        filteredCounters.computeIfAbsent(key, ignored -> Counter.builder("inference.routing.adapter.filtered")
                .description("Count of providers filtered out for adapter requests during routing")
                .tag("router", router)
                .tag("provider", providerId)
                .tag("model", modelId)
                .tag("tenant", tenantId)
                .tag("reason", reason)
                .tag("mode", distributionMode)
                .register(meterRegistry)).increment();
    }
}
