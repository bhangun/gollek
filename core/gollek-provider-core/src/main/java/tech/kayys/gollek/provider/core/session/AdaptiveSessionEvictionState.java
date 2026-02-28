package tech.kayys.gollek.provider.core.session;

/**
 * Mutable state for adaptive eviction policy telemetry.
 */
public final class AdaptiveSessionEvictionState {
    private double pressureEwma;
    private double reclaimFailureEwma;

    public synchronized double pressureEwma() {
        return pressureEwma;
    }

    public synchronized double reclaimFailureEwma() {
        return reclaimFailureEwma;
    }

    synchronized void update(double pressureEwma, double reclaimFailureEwma) {
        this.pressureEwma = pressureEwma;
        this.reclaimFailureEwma = reclaimFailureEwma;
    }
}

