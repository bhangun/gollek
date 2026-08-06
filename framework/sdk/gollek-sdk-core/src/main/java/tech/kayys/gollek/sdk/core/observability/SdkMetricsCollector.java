package tech.kayys.gollek.sdk.core.observability;

import java.time.Duration;

/**
 * Interface for clients to hook into SDK-level metrics natively.
 * Allows Wayang or other embedded clients to track TTFT and TPOT 
 * independently of the inference server.
 */
public interface SdkMetricsCollector {
    
    void recordClientSuccess(String model, long durationMs, int outputTokens);
    void recordClientFailure(String model, String errorType);
    
    void recordClientTtft(String model, long ttftMs);
    void recordClientTpot(String model, long tpotMs);
    void recordClientChunk(String model);
}
