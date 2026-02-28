package tech.kayys.gollek.provider.core.session;

import jakarta.enterprise.context.ApplicationScoped;

/**
 * EWMA-based adaptive eviction policy shared by all providers.
 */
@ApplicationScoped
public class EwmaAdaptiveSessionEvictionPolicy implements AdaptiveSessionEvictionPolicy {

    public static final EwmaAdaptiveSessionEvictionPolicy DEFAULT = new EwmaAdaptiveSessionEvictionPolicy();

    private static final double ALPHA = 0.20d;

    @Override
    public int resolveIdleTimeoutSeconds(AdaptiveSessionEvictionState state, int baseIdleTimeoutSeconds, double utilization) {
        int base = Math.max(1, baseIdleTimeoutSeconds);
        double pressure = pressureScore(state);

        if (pressure >= 0.85d) {
            return Math.max(10, base / 5);
        }
        if (utilization >= 0.90d || pressure >= 0.60d) {
            return Math.max(15, base / 4);
        }
        if (utilization >= 0.75d || pressure >= 0.35d) {
            return Math.max(20, base / 2);
        }
        return base;
    }

    @Override
    public void recordTelemetry(AdaptiveSessionEvictionState state, boolean underPressure, int reclaimedSessions) {
        double currentPressure = state.pressureEwma();
        double currentReclaimFailure = state.reclaimFailureEwma();

        double nextPressure = ewma(currentPressure, underPressure ? 1.0d : 0.0d);
        double nextReclaimFailure;
        if (underPressure) {
            nextReclaimFailure = ewma(currentReclaimFailure, reclaimedSessions > 0 ? 0.0d : 1.0d);
        } else {
            nextReclaimFailure = ewma(currentReclaimFailure, 0.0d);
        }

        state.update(nextPressure, nextReclaimFailure);
    }

    @Override
    public double pressureScore(AdaptiveSessionEvictionState state) {
        return clamp((0.6d * state.pressureEwma()) + (0.4d * state.reclaimFailureEwma()));
    }

    private double ewma(double current, double sample) {
        return current + (ALPHA * (sample - current));
    }

    private double clamp(double value) {
        if (value < 0.0d) {
            return 0.0d;
        }
        if (value > 1.0d) {
            return 1.0d;
        }
        return value;
    }
}

