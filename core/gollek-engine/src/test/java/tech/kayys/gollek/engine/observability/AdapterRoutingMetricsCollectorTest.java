package tech.kayys.gollek.engine.observability;

import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class AdapterRoutingMetricsCollectorTest {

    @Test
    void incrementsFilteredCounterWithStableTags() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        AdapterRoutingMetricsCollector collector = new AdapterRoutingMetricsCollector(registry);

        collector.recordProviderFiltered(
                "model-router-service",
                "litert-provider",
                "llama-3",
                "community",
                "adapter_unsupported");
        collector.recordProviderFiltered(
                "model-router-service",
                "litert-provider",
                "llama-3",
                "community",
                "adapter_unsupported");

        double count = registry.get("inference.routing.adapter.filtered")
                .tag("router", "model-router-service")
                .tag("provider", "litert-provider")
                .tag("model", "llama-3")
                .tag("tenant", "community")
                .tag("reason", "adapter_unsupported")
                .tag("mode", "standalone")
                .counter()
                .count();

        assertEquals(2.0, count);
    }
}
