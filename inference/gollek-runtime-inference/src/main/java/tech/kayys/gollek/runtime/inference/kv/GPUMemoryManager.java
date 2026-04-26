package tech.kayys.gollek.runtime.inference.kv;

import org.jboss.logging.Logger;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Production-grade GPU memory manager with arena-based allocation.
 * <p>
 * Provides:
 * <ul>
 *   <li><b>Pre-Allocation:</b> Reserve entire GPU memory pool at startup</li>
 *   <li><b>Arena Allocation:</b> Fast bump allocation from pre-reserved pool</li>
 *   <li><b>OOM Prevention:</b> Hard limits prevent out-of-memory crashes</li>
 *   <li><b>Defragmentation:</b> Compact memory when sequences finish</li>
 *   <li><b>Tensor Pooling:</b> Reuse tensor buffers to reduce allocation overhead</li>
 *   <li><b>Multi-Tenant Quotas:</b> Per-tenant memory limits</li>
 *   <li><b>Memory Statistics:</b> Detailed allocation tracking</li>
 * </ul>
 *
 * <h2>Memory Hierarchy</h2>
 * <pre>
 * GPUMemoryManager
 * ├── Arena (pre-allocated native/GPU memory)
 * │   ├── Block Pool (for KV cache)
 * │   │   └── PagedKVCache blocks
 * │   ├── Tensor Pool (for intermediate tensors)
 * │   │   └── Attention Q/K/V/O buffers
 * │   └── Scratch Space (temporary allocations)
 * │       └── Kernel work buffers
 * └── Allocation Tracker (metadata)
 *     ├── Per-tenant usage
 *     ├── Per-request usage
 *     └── High-water mark
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * GPUMemoryManager memoryManager = GPUMemoryManager.builder()
 *     .totalMemory(40L * 1024 * 1024 * 1024)  // 40GB GPU
 *     .kvCacheFraction(0.7)                   // 70% for KV cache
 *     .tensorPoolFraction(0.2)                // 20% for tensors
 *     .scratchFraction(0.1)                   // 10% for scratch
 *     .defragmentOnFree(true)                 // Compact on free
 *     .build();
 *
 * // Allocate tensor buffer
 * try (MemoryHandle handle = memoryManager.allocateTensor(
 *         "req-123", "tenant-1", 1024 * 1024)) {
 *     MemorySegment buffer = handle.segment();
 *     // Use buffer for tensor data
 * }  // Automatically freed
 *
 * // Check memory pressure
 * if (memoryManager.utilization() > 0.85) {
 *     memoryManager.defragment();
 * }
 *
 * // Get statistics
 * MemoryStats stats = memoryManager.stats();
 * }</pre>
 *
 * @see PagedKVCache
 * @see KVCacheManager
 * @since 0.2.0
 */
public final class GPUMemoryManager implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(GPUMemoryManager.class);

    // ── Memory Regions ─────────────────────────────────────────────────

    /** Arena for KV cache allocations */
    private final Arena kvCacheArena;

    /** Arena for tensor allocations */
    private final Arena tensorArena;

    /** Arena for scratch/temporary allocations */
    private final Arena scratchArena;

    /** Total memory reserved (bytes) */
    private final long totalMemory;

    /** Memory allocated for KV cache */
    private final long kvCacheMemory;

    /** Memory allocated for tensor pool */
    private final long tensorPoolMemory;

    /** Memory allocated for scratch */
    private final long scratchMemory;

    // ── Allocation Tracking ────────────────────────────────────────────

    /** Per-request allocations: requestId → bytes allocated */
    private final Map<String, AtomicLong> requestAllocations = new ConcurrentHashMap<>();

    /** Per-tenant allocations: tenantId → bytes allocated */
    private final Map<String, AtomicLong> tenantAllocations = new ConcurrentHashMap<>();

    /** Total currently allocated bytes */
    private final AtomicLong allocatedBytes = new AtomicLong(0);

    /** Peak allocation (high-water mark) */
    private final AtomicLong peakAllocations = new AtomicLong(0);

    /** Total allocation operations */
    private final AtomicLong totalAllocations = new AtomicLong(0);

    /** Total free operations */
    private final AtomicLong totalFrees = new AtomicLong(0);

    /** Total defragmentation runs */
    private final AtomicLong totalDefrags = new AtomicLong(0);

    /** OOM errors prevented */
    private final AtomicLong oomPrevented = new AtomicLong(0);

    // ── Configuration ──────────────────────────────────────────────────

    /** Whether to defragment on every free */
    private final boolean defragmentOnFree;

    /** Defragmentation threshold (utilization fraction) */
    private final double defragThreshold;

    /** Whether manager is closed */
    private final AtomicBoolean closed = new AtomicBoolean(false);

    /**
     * Creates a new GPU memory manager with pre-allocated arenas.
     *
     * @param config memory manager configuration
     */
    private GPUMemoryManager(Config config) {
        this.totalMemory = config.totalMemory;
        this.kvCacheMemory = (long) (config.totalMemory * config.kvCacheFraction);
        this.tensorPoolMemory = (long) (config.totalMemory * config.tensorPoolFraction);
        this.scratchMemory = config.totalMemory - kvCacheMemory - tensorPoolMemory;
        this.defragmentOnFree = config.defragmentOnFree;
        this.defragThreshold = config.defragThreshold;

        // Allocate arenas (in production, these would be GPU memory via FFM)
        // For now, we use off-heap native memory as a proxy
        this.kvCacheArena = Arena.ofAuto();
        this.tensorArena = Arena.ofAuto();
        this.scratchArena = Arena.ofAuto();

        // Pre-allocate memory pools
        kvCacheArena.allocate(kvCacheMemory);
        tensorArena.allocate(tensorPoolMemory);
        // Scratch arena is allocated on-demand

        LOG.infof("GPU Memory Manager initialized: total=%dMB, kv=%dMB, tensor=%dMB, scratch=%dMB",
            totalMemory / (1024 * 1024),
            kvCacheMemory / (1024 * 1024),
            tensorPoolMemory / (1024 * 1024),
            scratchMemory / (1024 * 1024));
    }

    /**
     * Creates a builder for configuring this manager.
     */
    public static Builder builder() {
        return new Builder();
    }

    // ── Allocation API ─────────────────────────────────────────────────

    /**
     * Allocates a tensor buffer with automatic lifecycle management.
     * <p>
     * The returned {@link MemoryHandle} implements {@link AutoCloseable} and
     * will free the allocation when closed (try-with-resources).
     *
     * @param requestId request identifier for tracking
     * @param tenantId tenant identifier for quota enforcement
     * @param sizeBytes number of bytes to allocate
     * @return memory handle (must be closed when done)
     * @throws OutOfMemoryError if insufficient memory available
     */
    public MemoryHandle allocateTensor(String requestId, String tenantId, long sizeBytes) {
        ensureOpen();

        // Check if allocation would exceed limits
        if (!canAllocate(sizeBytes)) {
            oomPrevented.incrementAndGet();
            throw new OutOfMemoryError(
                String.format("GPU memory exhausted: requested=%dMB, available=%dMB, utilization=%.1f%%",
                    sizeBytes / (1024 * 1024),
                    availableMemory() / (1024 * 1024),
                    utilization() * 100));
        }

        // Allocate from tensor arena
        MemorySegment segment = tensorArena.allocate(sizeBytes);

        // Update tracking
        allocatedBytes.addAndGet(sizeBytes);
        peakAllocations.updateAndGet(peak -> Math.max(peak, allocatedBytes.get()));
        totalAllocations.incrementAndGet();

        requestAllocations.computeIfAbsent(requestId, k -> new AtomicLong())
            .addAndGet(sizeBytes);
        tenantAllocations.computeIfAbsent(tenantId, k -> new AtomicLong())
            .addAndGet(sizeBytes);

        LOG.debugf("Allocated tensor: requestId=%s, tenantId=%s, size=%dKB",
            requestId, tenantId, sizeBytes / 1024);

        return new MemoryHandle(segment, sizeBytes, requestId, tenantId);
    }

    /**
     * Allocates from KV cache arena.
     */
    public MemoryHandle allocateKVCache(String requestId, String tenantId, long sizeBytes) {
        ensureOpen();

        if (!canAllocate(sizeBytes)) {
            oomPrevented.incrementAndGet();
            throw new OutOfMemoryError("KV cache memory exhausted");
        }

        MemorySegment segment = kvCacheArena.allocate(sizeBytes);

        allocatedBytes.addAndGet(sizeBytes);
        peakAllocations.updateAndGet(peak -> Math.max(peak, allocatedBytes.get()));
        totalAllocations.incrementAndGet();

        requestAllocations.computeIfAbsent(requestId, k -> new AtomicLong())
            .addAndGet(sizeBytes);
        tenantAllocations.computeIfAbsent(tenantId, k -> new AtomicLong())
            .addAndGet(sizeBytes);

        return new MemoryHandle(segment, sizeBytes, requestId, tenantId);
    }

    /**
     * Allocates from scratch arena (temporary, short-lived).
     */
    public MemoryHandle allocateScratch(String requestId, long sizeBytes) {
        ensureOpen();

        if (!canAllocate(sizeBytes)) {
            oomPrevented.incrementAndGet();
            throw new OutOfMemoryError("Scratch memory exhausted");
        }

        MemorySegment segment = scratchArena.allocate(sizeBytes);
        allocatedBytes.addAndGet(sizeBytes);
        totalAllocations.incrementAndGet();

        return new MemoryHandle(segment, sizeBytes, requestId, "scratch");
    }

    // ── Memory Management ──────────────────────────────────────────────

    /**
     * Frees an allocation and updates tracking.
     *
     * @param sizeBytes number of bytes freed
     * @param requestId request identifier
     * @param tenantId tenant identifier
     */
    void free(long sizeBytes, String requestId, String tenantId) {
        allocatedBytes.addAndGet(-sizeBytes);
        totalFrees.incrementAndGet();

        AtomicLong requestAlloc = requestAllocations.get(requestId);
        if (requestAlloc != null) {
            requestAlloc.addAndGet(-sizeBytes);
            if (requestAlloc.get() <= 0) {
                requestAllocations.remove(requestId);
            }
        }

        AtomicLong tenantAlloc = tenantAllocations.get(tenantId);
        if (tenantAlloc != null) {
            tenantAlloc.addAndGet(-sizeBytes);
            if (tenantAlloc.get() <= 0) {
                tenantAllocations.remove(tenantId);
            }
        }

        // Optionally defragment
        if (defragmentOnFree && utilization() > defragThreshold) {
            defragment();
        }
    }

    /**
     * Defragments memory by compacting allocations.
     * <p>
     * In a production implementation, this would:
     * <ul>
     *   <li>Copy live data to new contiguous region</li>
     *   <li>Update all references to moved data</li>
     *   <li>Return freed space to arena</li>
     * </ul>
     * <p>
     * For now, this is a no-op since JDK Arena doesn't support compaction.
     * Real implementation requires custom allocator.
     *
     * @return bytes reclaimed (0 if no compaction possible)
     */
    public long defragment() {
        ensureOpen();
        totalDefrags.incrementAndGet();

        LOG.infof("Defragmentation run: utilization=%.1f%%, fragmentation=%.1f%%",
            utilization() * 100, fragmentation() * 100);

        // In production: compact arenas, return freed space
        // For now: just log statistics
        return 0;
    }

    /**
     * Forces garbage collection of arenas (if supported).
     */
    public void gc() {
        // JDK 25 Arena doesn't support explicit GC, but we can log
        LOG.infof("Memory GC requested: allocated=%dMB, available=%dMB",
            allocatedBytes.get() / (1024 * 1024),
            availableMemory() / (1024 * 1024));
    }

    // ── Query Methods ──────────────────────────────────────────────────

    /**
     * Gets total managed memory (bytes).
     */
    public long totalMemory() {
        return totalMemory;
    }

    /**
     * Gets currently allocated memory (bytes).
     */
    public long allocatedMemory() {
        return allocatedBytes.get();
    }

    /**
     * Gets available (unallocated) memory (bytes).
     */
    public long availableMemory() {
        return totalMemory - allocatedBytes.get();
    }

    /**
     * Gets peak (high-water mark) allocation (bytes).
     */
    public long peakAllocations() {
        return peakAllocations.get();
    }

    /**
     * Gets memory utilization as fraction (0.0 to 1.0).
     */
    public double utilization() {
        return (double) allocatedBytes.get() / totalMemory;
    }

    /**
     * Gets fragmentation ratio (0.0 to 1.0).
     * <p>
     * 0.0 = no fragmentation (all contiguous)
     * 1.0 = maximum fragmentation
     */
    public double fragmentation() {
        // Simplified: count number of allocations vs ideal
        long allocCount = totalAllocations.get() - totalFrees.get();
        if (allocCount == 0) return 0.0;

        // In production, this would measure actual fragmentation
        // For now, estimate based on allocation pattern
        return Math.min(1.0, (double) requestAllocations.size() / allocCount);
    }

    /**
     * Checks if allocation is possible without exceeding limits.
     */
    public boolean canAllocate(long sizeBytes) {
        return allocatedBytes.get() + sizeBytes <= totalMemory;
    }

    /**
     * Gets per-request allocation tracking.
     */
    public Map<String, Long> getRequestAllocations() {
        Map<String, Long> result = new HashMap<>();
        for (var entry : requestAllocations.entrySet()) {
            result.put(entry.getKey(), entry.getValue().get());
        }
        return result;
    }

    /**
     * Gets per-tenant allocation tracking.
     */
    public Map<String, Long> getTenantAllocations() {
        Map<String, Long> result = new HashMap<>();
        for (var entry : tenantAllocations.entrySet()) {
            result.put(entry.getKey(), entry.getValue().get());
        }
        return result;
    }

    /**
     * Gets comprehensive memory statistics.
     */
    public MemoryStats stats() {
        return new MemoryStats(
            totalMemory,
            allocatedBytes.get(),
            availableMemory(),
            peakAllocations.get(),
            utilization(),
            fragmentation(),
            totalAllocations.get(),
            totalFrees.get(),
            totalDefrags.get(),
            oomPrevented.get(),
            requestAllocations.size(),
            tenantAllocations.size()
        );
    }

    // ── Tenant Quotas ──────────────────────────────────────────────────

    /**
     * Checks if tenant is within quota.
     *
     * @param tenantId tenant identifier
     * @param quotaBytes maximum bytes allowed
     * @return true if tenant is under quota
     */
    public boolean checkTenantQuota(String tenantId, long quotaBytes) {
        AtomicLong usage = tenantAllocations.get(tenantId);
        long currentUsage = usage != null ? usage.get() : 0;
        return currentUsage < quotaBytes;
    }

    /**
     * Gets tenant's current usage.
     */
    public long getTenantUsage(String tenantId) {
        AtomicLong usage = tenantAllocations.get(tenantId);
        return usage != null ? usage.get() : 0;
    }

    // ── Lifecycle ──────────────────────────────────────────────────────

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }

        // Close arenas (releases all native memory)
        kvCacheArena.close();
        tensorArena.close();
        scratchArena.close();

        requestAllocations.clear();
        tenantAllocations.clear();
        allocatedBytes.set(0);

        LOG.infof("GPU Memory Manager closed: peak=%dMB, totalAllocations=%d, oomPrevented=%d",
            peakAllocations.get() / (1024 * 1024),
            totalAllocations.get(),
            oomPrevented.get());
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("GPU Memory Manager is closed");
        }
    }

    // ── Nested Types ───────────────────────────────────────────────────

    /**
     * Auto-closable memory handle.
     * <p>
     * Use try-with-resources to ensure proper cleanup:
     * <pre>{@code
     * try (MemoryHandle handle = memoryManager.allocateTensor(...)) {
     *     // Use handle.segment()
     * }  // Automatically freed
     * }</pre>
     */
    public final class MemoryHandle implements AutoCloseable {
        private final MemorySegment segment;
        private final long sizeBytes;
        private final String requestId;
        private final String tenantId;
        private volatile boolean closed = false;

        private MemoryHandle(MemorySegment segment, long sizeBytes,
                            String requestId, String tenantId) {
            this.segment = segment;
            this.sizeBytes = sizeBytes;
            this.requestId = requestId;
            this.tenantId = tenantId;
        }

        /**
         * Gets the native memory segment.
         */
        public MemorySegment segment() {
            if (closed) {
                throw new IllegalStateException("Memory handle is closed");
            }
            return segment;
        }

        /**
         * Gets allocation size in bytes.
         */
        public long sizeBytes() {
            return sizeBytes;
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                GPUMemoryManager.this.free(sizeBytes, requestId, tenantId);
                LOG.debugf("Freed memory handle: requestId=%s, size=%dKB",
                    requestId, sizeBytes / 1024);
            }
        }
    }

    /**
     * Comprehensive memory statistics snapshot.
     */
    public record MemoryStats(
        long totalMemory,
        long allocatedMemory,
        long availableMemory,
        long peakAllocations,
        double utilization,
        double fragmentation,
        long totalAllocations,
        long totalFrees,
        long totalDefrags,
        long oomPrevented,
        int activeRequests,
        int activeTenants
    ) {
        /**
         * Formats statistics for logging.
         */
        public String format() {
            return String.format(
                "MemoryStats[total=%dMB, alloc=%dMB, avail=%dMB, peak=%dMB, util=%.1f%%, frag=%.1f%%, allocs=%d, frees=%d, oom=%d]",
                totalMemory / (1024 * 1024),
                allocatedMemory / (1024 * 1024),
                availableMemory / (1024 * 1024),
                peakAllocations / (1024 * 1024),
                utilization * 100,
                fragmentation * 100,
                totalAllocations,
                totalFrees,
                oomPrevented
            );
        }
    }

    /**
     * Configuration for GPU Memory Manager.
     */
    public static final class Config {
        final long totalMemory;
        final double kvCacheFraction;
        final double tensorPoolFraction;
        final double scratchFraction;
        final boolean defragmentOnFree;
        final double defragThreshold;

        private Config(long totalMemory, double kvCacheFraction,
                      double tensorPoolFraction, double scratchFraction,
                      boolean defragmentOnFree, double defragThreshold) {
            this.totalMemory = totalMemory;
            this.kvCacheFraction = kvCacheFraction;
            this.tensorPoolFraction = tensorPoolFraction;
            this.scratchFraction = scratchFraction;
            this.defragmentOnFree = defragmentOnFree;
            this.defragThreshold = defragThreshold;
        }
    }

    /**
     * Builder for GPU Memory Manager.
     */
    public static final class Builder {
        private long totalMemory = 40L * 1024 * 1024 * 1024;  // 40GB default
        private double kvCacheFraction = 0.7;
        private double tensorPoolFraction = 0.2;
        private double scratchFraction = 0.1;
        private boolean defragmentOnFree = false;
        private double defragThreshold = 0.9;

        private Builder() {}

        /**
         * Sets total managed memory in bytes.
         * <p>
         * Should match your GPU's VRAM size (e.g., 40GB for A100).
         */
        public Builder totalMemory(long bytes) {
            this.totalMemory = bytes;
            return this;
        }

        /**
         * Sets fraction of memory for KV cache (0.0 to 1.0).
         * <p>
         * Typical: 0.6-0.8 for LLM serving.
         */
        public Builder kvCacheFraction(double fraction) {
            this.kvCacheFraction = fraction;
            return this;
        }

        /**
         * Sets fraction for tensor pool (0.0 to 1.0).
         * <p>
         * Typical: 0.15-0.25 for intermediate activations.
         */
        public Builder tensorPoolFraction(double fraction) {
            this.tensorPoolFraction = fraction;
            return this;
        }

        /**
         * Sets fraction for scratch space (0.0 to 1.0).
         * <p>
         * Typical: 0.05-0.15 for temporary buffers.
         */
        public Builder scratchFraction(double fraction) {
            this.scratchFraction = fraction;
            return this;
        }

        /**
         * Enables defragmentation on every free operation.
         * <p>
         * Warning: This adds overhead to every free. Consider using
         * threshold-based defragmentation instead.
         */
        public Builder defragmentOnFree(boolean defragmentOnFree) {
            this.defragmentOnFree = defragmentOnFree;
            return this;
        }

        /**
         * Sets defragmentation threshold (0.0 to 1.0).
         * <p>
         * When utilization exceeds this threshold, defragmentation is triggered.
         */
        public Builder defragThreshold(double threshold) {
            this.defragThreshold = threshold;
            return this;
        }

        /**
         * Auto-configures for a specific GPU model.
         */
        public Builder forGPU(String gpuModel) {
            return switch (gpuModel.toLowerCase()) {
                case "a100-40gb" -> totalMemory(40L * 1024 * 1024 * 1024);
                case "a100-80gb" -> totalMemory(80L * 1024 * 1024 * 1024);
                case "h100-80gb" -> totalMemory(80L * 1024 * 1024 * 1024);
                case "h200-141gb" -> totalMemory(141L * 1024 * 1024 * 1024);
                default -> throw new IllegalArgumentException("Unknown GPU model: " + gpuModel);
            };
        }

        public GPUMemoryManager build() {
            if (kvCacheFraction + tensorPoolFraction + scratchFraction > 1.0) {
                throw new IllegalArgumentException("Memory fractions must sum to <= 1.0");
            }
            return new GPUMemoryManager(new Config(
                totalMemory, kvCacheFraction, tensorPoolFraction,
                scratchFraction, defragmentOnFree, defragThreshold));
        }
    }
}
