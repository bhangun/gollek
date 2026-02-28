package tech.kayys.gollek.provider.core.session;

/**
 * Provider-agnostic policy contract for adaptive idle-session eviction.
 * Implementations can use runtime telemetry to tighten or relax idle timeout.
 */
public interface AdaptiveSessionEvictionPolicy {

    /**
     * Resolve idle-timeout (seconds) using pool utilization and telemetry state.
     */
    int resolveIdleTimeoutSeconds(AdaptiveSessionEvictionState state, int baseIdleTimeoutSeconds, double utilization);

    /**
     * Record telemetry sample from a cleanup/reclaim cycle.
     */
    void recordTelemetry(AdaptiveSessionEvictionState state, boolean underPressure, int reclaimedSessions);

    /**
     * Current pressure score in [0.0, 1.0].
     */
    double pressureScore(AdaptiveSessionEvictionState state);
}

