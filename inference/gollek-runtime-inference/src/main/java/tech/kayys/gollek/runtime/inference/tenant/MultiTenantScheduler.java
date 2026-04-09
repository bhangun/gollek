package tech.kayys.gollek.runtime.inference.tenant;

import tech.kayys.gollek.runtime.inference.batch.BatchRequest;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Multi-tenant scheduler using Deficit Round Robin (DRR) for fair
 * compute allocation across tenants with different service tiers.
 * <p>
 * DRR ensures that:
 * <ul>
 *   <li>ENTERPRISE tenants get 10x the compute of FREE tenants</li>
 *   <li>No tenant starves — every tenant eventually gets scheduled</li>
 *   <li>Latency-sensitive requests can be prioritized via tier</li>
 * </ul>
 * <p>
 * This is the real control plane of the Gollek runtime, making it
 * production-grade for multi-tenant deployments.
 */
public final class MultiTenantScheduler {

    private final Map<String, TenantContext> tenants = new ConcurrentHashMap<>();
    private final Map<String, Integer> deficits = new ConcurrentHashMap<>();
    private final List<String> roundRobin = new ArrayList<>();

    /** Register a new tenant. */
    public void registerTenant(TenantContext tenant) {
        tenants.put(tenant.tenantId, tenant);
        deficits.put(tenant.tenantId, 0);
        if (!roundRobin.contains(tenant.tenantId)) {
            roundRobin.add(tenant.tenantId);
        }
    }

    /** Remove a tenant. */
    public void unregisterTenant(String tenantId) {
        tenants.remove(tenantId);
        deficits.remove(tenantId);
        roundRobin.remove(tenantId);
    }

    /** Get a tenant's context. */
    public TenantContext getTenant(String tenantId) {
        return tenants.get(tenantId);
    }

    /**
     * Submit a request to the appropriate tenant's decode queue.
     *
     * @return true if accepted, false if rejected (quota exceeded)
     */
    public boolean submitDecode(BatchRequest request) {
        TenantContext tenant = tenants.get(request.tenantId);
        if (tenant == null) return false;

        if (!tenant.canAcceptRequest()) {
            return false; // backpressure
        }

        tenant.activeRequests.incrementAndGet();
        tenant.decodeQueue.add(request);
        return true;
    }

    /**
     * Build the next decode batch using DRR fair scheduling.
     *
     * @param maxBatchSize maximum batch size
     * @return list of requests to execute in the next step
     */
    public List<BatchRequest> nextDecodeBatch(int maxBatchSize) {
        List<BatchRequest> batch = new ArrayList<>();

        for (String tenantId : roundRobin) {
            TenantContext tenant = tenants.get(tenantId);
            if (tenant == null) continue;

            int quantum = tenant.tier.quantum();
            int deficit = deficits.getOrDefault(tenantId, 0) + quantum;

            while (deficit > 0 && !tenant.decodeQueue.isEmpty()) {
                BatchRequest r = tenant.decodeQueue.peek();
                int cost = 1; // 1 credit per request per step

                if (cost > deficit) break;

                tenant.decodeQueue.poll();
                batch.add(r);
                deficit -= cost;

                if (batch.size() >= maxBatchSize) {
                    deficits.put(tenantId, deficit);
                    return batch;
                }
            }

            deficits.put(tenantId, Math.max(0, deficit));
        }

        return batch;
    }

    /**
     * Mark a request as completed, releasing tenant resources.
     */
    public void complete(BatchRequest request) {
        TenantContext tenant = tenants.get(request.tenantId);
        if (tenant != null) {
            tenant.activeRequests.decrementAndGet();
        }
    }

    /** Number of registered tenants. */
    public int tenantCount() {
        return tenants.size();
    }
}
