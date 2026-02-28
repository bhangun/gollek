package tech.kayys.gollek.spi.observability;

import java.time.Duration;

/**
 * Default no-op implementation for runtimes that do not wire metric backends.
 */
public class NoopAdapterMetricsRecorder implements AdapterMetricsRecorder {

    @Override
    public void recordRequest(String provider, String adapterType) {
    }

    @Override
    public void recordSessionAcquire(String provider, String adapterType, Duration duration) {
    }

    @Override
    public void recordInitWait(String provider, Duration duration) {
    }

    @Override
    public void recordApply(String provider, String adapterType, Duration duration) {
    }

    @Override
    public void recordCacheHit(String provider, String scope) {
    }

    @Override
    public void recordCacheMiss(String provider, String scope) {
    }
}
