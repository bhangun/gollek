/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

/**
 * Global "Virtual Memory Manager" for LLM tokens.
 * Coordinates KV cache allocation, reuse, and eviction across all runners.
 * Enhanced for cluster-wide distributed memory and pluggable attention.
 */
@ApplicationScoped
public class KVMemoryManager {

    private final Map<KVKey, UnifiedKVCache> memoryHandle = new ConcurrentHashMap<>();

    @Inject
    KVDirectory directory;

    @Inject
    AttentionKernelRegistry kernelRegistry;

    @Inject
    RemoteAttentionExecutor remoteExecutor;

    /**
     * Executes attention either locally or remotely based on bandwidth and location.
     * 
     * @param key the KV shard key
     * @param query the query tensor segment
     * @param context the attention context
     * @return a future resolving to the output tensor segment
     */
    public CompletableFuture<java.lang.foreign.MemorySegment> executeRemoteIfOptimal(
        KVKey key, 
        java.lang.foreign.MemorySegment query, 
        AttentionKernel.AttentionContext context
    ) {
        UnifiedKVCache local = get(key);
        if (local != null) {
            // Compute locally if we have the data
            // (Placeholder: actual local kernel call)
            return CompletableFuture.completedFuture(null);
        }

        return locateRemote(key).thenCompose(locations -> {
            if (locations.isEmpty()) {
                return CompletableFuture.failedFuture(new RuntimeException("KV shard not found: " + key));
            }

            KVLocation best = locations.get(0);
            if (best.nodeId().equals("local")) {
                return CompletableFuture.completedFuture(null);
            }

            // BANDWIDTH DECISION: 
            // If KV is huge but query is small -> send query to data node
            return remoteExecutor.executeRemote(key, query, best, context);
        });
    }

    /**
     * Retrieves an existing cache or creates a new one using the provided supplier.
     */
    public UnifiedKVCache getOrCreate(KVKey key, Supplier<UnifiedKVCache> creator) {
        return memoryHandle.computeIfAbsent(key, k -> {
            UnifiedKVCache cache = creator.get();
            // Register in the cluster directory
            directory.register(key, KVLocation.local("localhost:9090")); 
            return cache;
        });
    }

    /**
     * Locates a KV shard across the cluster.
     */
    public CompletableFuture<List<KVLocation>> locateRemote(KVKey key) {
        return directory.locate(key);
    }

    /**
     * Prefetches a remote KV shard into the local memory fabric.
     */
    public CompletableFuture<UnifiedKVCache> prefetch(KVKey key) {
        return locateRemote(key).thenCompose(locations -> {
            if (locations.isEmpty()) {
                return CompletableFuture.failedFuture(new RuntimeException("Shard not found: " + key));
            }
            // TODO: Implement actual transport (gRPC/RDMA) to fetch the segment
            return CompletableFuture.completedFuture(get(key)); 
        });
    }

    /**
     * Resolves the hardware-optimal attention kernel for the current context.
     */
    public AttentionKernel getKernel(KernelType preferred) {
        return kernelRegistry.resolveOptimal(preferred);
    }

    /**
     * Manually registers a cache in the global fabric.
     */
    public void put(KVKey key, UnifiedKVCache cache) {
        memoryHandle.put(key, cache);
        directory.register(key, KVLocation.local("localhost:9090"));
    }

    /**
     * Gets a cache if it exists.
     */
    public UnifiedKVCache get(KVKey key) {
        return memoryHandle.get(key);
    }

    /**
     * Moves or shares a cache between different lookup keys.
     */
    public void share(KVKey from, KVKey to) {
        UnifiedKVCache cache = memoryHandle.get(from);
        if (cache != null) {
            memoryHandle.put(to, cache);
        }
    }

    /**
     * Removes a cache from memory.
     */
    public void evict(KVKey key) {
        UnifiedKVCache cache = memoryHandle.remove(key);
        if (cache != null) {
            cache.close();
        }
    }

    /**
     * Proactively fetches MoE expert weights into the local execution node.
     * Unlike KV cache, expert weights are static but massive (GBs), 
     * requiring affinity-based loading.
     */
    public CompletableFuture<Void> prefetchExperts(String modelId, java.util.List<String> expertIds) {
        // TODO: Implement expert affinity lookup and parallel loading
        return CompletableFuture.completedFuture(null);
    }

    /**
     * Spills the cache to persistent storage (mmap/disk).

    /**
     * Current memory statistics.
     */
    public long getCacheCount() {
        return memoryHandle.size();
    }
}
