package tech.kayys.gollek.runtime.inference.tenant;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * KV cache block quota manager for multi-tenant scheduling.
 * <p>
 * Prevents any single tenant from consuming all KV cache memory,
 * which would cause system-wide collapse under contention.
 */
public final class KVQuotaManager {

    private final MultiTenantScheduler scheduler;

    public KVQuotaManager(MultiTenantScheduler scheduler) {
        this.scheduler = scheduler;
    }

    /**
     * Try to allocate KV cache blocks for a tenant.
     *
     * @param tenantId tenant identifier
     * @param blocks   number of blocks to allocate
     * @return true if allocation succeeded, false if quota exceeded
     */
    public boolean tryAllocate(String tenantId, int blocks) {
        TenantContext tenant = scheduler.getTenant(tenantId);
        if (tenant == null) return false;

        if (!tenant.canAllocateBlocks(blocks)) {
            return false;
        }

        tenant.usedBlocks.addAndGet(blocks);
        return true;
    }

    /**
     * Release KV cache blocks for a tenant.
     *
     * @param tenantId tenant identifier
     * @param blocks   number of blocks to release
     */
    public void release(String tenantId, int blocks) {
        TenantContext tenant = scheduler.getTenant(tenantId);
        if (tenant != null) {
            tenant.usedBlocks.addAndGet(-blocks);
        }
    }

    /**
     * Get current KV block usage for a tenant.
     */
    public int usage(String tenantId) {
        TenantContext tenant = scheduler.getTenant(tenantId);
        return tenant != null ? tenant.usedBlocks.get() : 0;
    }
}
