package tech.kayys.gollek.sdk.core.observability;

/**
 * Interface for clients to hook into SDK-level audit trailing.
 */
public interface SdkAuditLogger {
    void logClientRequestStart(String requestId, String model);
    void logClientRequestComplete(String requestId, String model, int outputTokens, long durationMs);
    void logClientRequestFailure(String requestId, String model, String errorType);
}
