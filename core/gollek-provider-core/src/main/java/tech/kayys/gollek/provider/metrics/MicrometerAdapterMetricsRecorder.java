package tech.kayys.gollek.provider.metrics;

import io.micrometer.core.instrument.MeterRegistry;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.observability.AdapterMetricSchema;
import tech.kayys.gollek.spi.observability.AdapterMetricsRecorder;

import java.time.Duration;

/**
 * Shared Micrometer-backed adapter metric recorder for all providers.
 */
@ApplicationScoped
public class MicrometerAdapterMetricsRecorder implements AdapterMetricsRecorder {

    @Inject
    MeterRegistry meterRegistry;

    @Override
    public void recordRequest(String provider, String adapterType) {
        meterRegistry.counter(
                AdapterMetricSchema.REQUEST_TOTAL,
                AdapterMetricSchema.TAG_PROVIDER, provider,
                AdapterMetricSchema.TAG_TYPE, normalizeType(adapterType))
                .increment();
    }

    @Override
    public void recordSessionAcquire(String provider, String adapterType, Duration duration) {
        if (duration == null) {
            return;
        }
        meterRegistry.timer(
                AdapterMetricSchema.SESSION_ACQUIRE_DURATION,
                AdapterMetricSchema.TAG_PROVIDER, provider,
                AdapterMetricSchema.TAG_TYPE, normalizeType(adapterType))
                .record(duration);
    }

    @Override
    public void recordInitWait(String provider, Duration duration) {
        if (duration == null) {
            return;
        }
        meterRegistry.timer(
                AdapterMetricSchema.INIT_WAIT_DURATION,
                AdapterMetricSchema.TAG_PROVIDER, provider)
                .record(duration);
    }

    @Override
    public void recordApply(String provider, String adapterType, Duration duration) {
        if (duration == null) {
            return;
        }
        meterRegistry.timer(
                AdapterMetricSchema.APPLY_DURATION,
                AdapterMetricSchema.TAG_PROVIDER, provider,
                AdapterMetricSchema.TAG_TYPE, normalizeType(adapterType))
                .record(duration);
    }

    @Override
    public void recordCacheHit(String provider, String scope) {
        meterRegistry.counter(
                AdapterMetricSchema.CACHE_HIT,
                AdapterMetricSchema.TAG_PROVIDER, provider,
                AdapterMetricSchema.TAG_SCOPE, normalizeScope(scope))
                .increment();
    }

    @Override
    public void recordCacheMiss(String provider, String scope) {
        meterRegistry.counter(
                AdapterMetricSchema.CACHE_MISS,
                AdapterMetricSchema.TAG_PROVIDER, provider,
                AdapterMetricSchema.TAG_SCOPE, normalizeScope(scope))
                .increment();
    }

    private String normalizeType(String adapterType) {
        return adapterType == null || adapterType.isBlank() ? AdapterMetricSchema.TYPE_UNKNOWN : adapterType;
    }

    private String normalizeScope(String scope) {
        return scope == null || scope.isBlank() ? "default" : scope;
    }
}
