package tech.kayys.gollek.runtime.inference.tenant;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

import tech.kayys.gollek.runtime.inference.batch.BatchRequest;

/**
 * Per-tenant runtime context for multi-tenant scheduling.
 * <p>
 * Tracks quotas, active resources, and request queues per tenant.
 * The {@link MultiTenantScheduler} uses this to enforce fairness
 * and resource isolation.
 */
public final class TenantContext {

    /** Tenant identifier. */
    public final String tenantId;

    /** Service tier determining scheduling priority. */
    public final TenantTier tier;

    // ── Quotas ────────────────────────────────────────────────────────

    /** Maximum concurrent requests allowed for this tenant. */
    public final int maxConcurrentRequests;

    /** Maximum KV cache blocks this tenant can use. */
    public final int maxKvBlocks;

    /** Maximum tokens per second rate limit. */
    public final int maxTokensPerSecond;

    // ── Runtime state ─────────────────────────────────────────────────

    /** Number of currently active requests. */
    public final AtomicInteger activeRequests = new AtomicInteger(0);

    /** Number of KV cache blocks currently in use. */
    public final AtomicInteger usedBlocks = new AtomicInteger(0);

    /** Pending prefill requests. */
    public final Queue<BatchRequest> prefillQueue = new ConcurrentLinkedQueue<>();

    /** Pending decode requests. */
    public final Queue<BatchRequest> decodeQueue = new ConcurrentLinkedQueue<>();

    public TenantContext(String tenantId, TenantTier tier,
                          int maxConcurrentRequests, int maxKvBlocks,
                          int maxTokensPerSecond) {
        this.tenantId = tenantId;
        this.tier = tier;
        this.maxConcurrentRequests = maxConcurrentRequests;
        this.maxKvBlocks = maxKvBlocks;
        this.maxTokensPerSecond = maxTokensPerSecond;
    }

    /** Check if this tenant can accept another concurrent request. */
    public boolean canAcceptRequest() {
        return activeRequests.get() < maxConcurrentRequests;
    }

    /** Check if this tenant can allocate more KV cache blocks. */
    public boolean canAllocateBlocks(int blocks) {
        return usedBlocks.get() + blocks <= maxKvBlocks;
    }

    @Override
    public String toString() {
        return "TenantContext[id=" + tenantId
            + ", tier=" + tier
            + ", active=" + activeRequests.get()
            + ", kvBlocks=" + usedBlocks.get() + "/" + maxKvBlocks + "]";
    }
}
