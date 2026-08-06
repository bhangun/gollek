package tech.kayys.gollek.sdk.core.observability;

/**
 * Registry interface to wire in client-side observability implementations
 * like Datadog, Prometheus, or Wayang's native OpenTelemetry.
 */
public interface SdkObservabilityProvider {
    SdkMetricsCollector getMetricsCollector();
    SdkAuditLogger getAuditLogger();
    
    /**
     * A default no-op provider for when observability isn't needed.
     */
    static SdkObservabilityProvider noop() {
        return new SdkObservabilityProvider() {
            @Override
            public SdkMetricsCollector getMetricsCollector() {
                return new SdkMetricsCollector() {
                    public void recordClientSuccess(String m, long d, int t) {}
                    public void recordClientFailure(String m, String e) {}
                    public void recordClientTtft(String m, long t) {}
                    public void recordClientTpot(String m, long t) {}
                    public void recordClientChunk(String m) {}
                };
            }
            @Override
            public SdkAuditLogger getAuditLogger() {
                return new SdkAuditLogger() {
                    public void logClientRequestStart(String id, String m) {}
                    public void logClientRequestComplete(String id, String m, int t, long d) {}
                    public void logClientRequestFailure(String id, String m, String e) {}
                };
            }
        };
    }
}
