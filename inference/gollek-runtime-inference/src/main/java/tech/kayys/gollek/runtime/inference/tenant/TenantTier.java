package tech.kayys.gollek.runtime.inference.tenant;

/**
 * Service tier for multi-tenant scheduling priority.
 * <p>
 * Higher tiers receive more scheduling quantum (credits) in the
 * Deficit Round Robin scheduler, resulting in proportionally
 * more compute time and lower latency.
 */
public enum TenantTier {

    FREE(1),
    PRO(3),
    ENTERPRISE(10);

    private final int quantum;

    TenantTier(int quantum) {
        this.quantum = quantum;
    }

    /** Scheduling quantum (weight) for DRR. */
    public int quantum() {
        return quantum;
    }
}
