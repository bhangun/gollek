package tech.kayys.gollek.provider.metrics;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.observability.AdapterMetricSchema;
import tech.kayys.gollek.spi.observability.AdapterMetricsRecorder;

import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Shared Micrometer-backed adapter metric recorder for all providers.
 */
@ApplicationScoped
public class MicrometerAdapterMetricsRecorder implements AdapterMetricsRecorder {

    @Inject
    MeterRegistry meterRegistry;

    private static final String METRIC_PREFIX = "gollek.provider.";

    @Override
    public void recordSuccess(AdapterMetricSchema schema, long durationMs) {
        Tags tags = buildTags(schema);
        
        meterRegistry.counter(METRIC_PREFIX + "requests.total", tags).increment();
        meterRegistry.counter(METRIC_PREFIX + "requests.success", tags).increment();
        meterRegistry.timer(METRIC_PREFIX + "duration", tags)
                .record(durationMs, TimeUnit.MILLISECONDS);
    }

    @Override
    public void recordFailure(AdapterMetricSchema schema, long durationMs, String error) {
        Tags tags = buildTags(schema).and("error", error != null ? error : "unknown");
        
        meterRegistry.counter(METRIC_PREFIX + "requests.total", tags).increment();
        meterRegistry.counter(METRIC_PREFIX + "requests.failure", tags).increment();
        meterRegistry.timer(METRIC_PREFIX + "duration", tags)
                .record(durationMs, TimeUnit.MILLISECONDS);
    }

    @Override
    public void recordTokens(AdapterMetricSchema schema, int inputTokens, int outputTokens) {
        Tags tags = buildTags(schema);
        
        meterRegistry.counter(METRIC_PREFIX + "tokens.input", tags).increment(inputTokens);
        meterRegistry.counter(METRIC_PREFIX + "tokens.output", tags).increment(outputTokens);
        meterRegistry.counter(METRIC_PREFIX + "tokens.total", tags)
                .increment((double) inputTokens + outputTokens);
    }

    @Override
    public void close() {
        // No-op for Micrometer registry
    }

    private Tags buildTags(AdapterMetricSchema schema) {
        Tags tags = Tags.of(
                "adapter", schema.adapterId() != null ? schema.adapterId() : "unknown",
                "model", schema.modelId() != null ? schema.modelId() : "unknown",
                "operation", schema.operation() != null ? schema.operation() : "unknown"
        );
        
        if (schema.tags() != null) {
            for (Map.Entry<String, String> entry : schema.tags().entrySet()) {
                tags = tags.and(entry.getKey(), entry.getValue());
            }
        }
        
        return tags;
    }
}
