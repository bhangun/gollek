package tech.kayys.gollek.spi.observability;

import java.time.Duration;

/**
 * Provider-agnostic adapter metric recorder contract.
 */
public interface AdapterMetricsRecorder {

    void recordRequest(String provider, String adapterType);

    void recordSessionAcquire(String provider, String adapterType, Duration duration);

    void recordInitWait(String provider, Duration duration);

    void recordApply(String provider, String adapterType, Duration duration);

    void recordCacheHit(String provider, String scope);

    void recordCacheMiss(String provider, String scope);
}
