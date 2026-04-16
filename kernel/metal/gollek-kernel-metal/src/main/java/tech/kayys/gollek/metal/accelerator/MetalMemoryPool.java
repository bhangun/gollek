package tech.kayys.gollek.multimodal.metal;

import org.jboss.logging.Logger;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Metal memory pool for Apple Silicon GPU optimization.
 * Manages unified memory for efficient CPU/GPU data sharing.
 */
public class MetalMemoryPool {

    private static final Logger log = Logger.getLogger(MetalMemoryPool.class);

    private final long blockSize;
    private final int maxBlocks;
    private final ConcurrentHashMap<Long, MemorySegment> freeBlocks;
    private final ConcurrentHashMap<Long, MemorySegment> allocatedBlocks;
    private final AtomicLong totalAllocations;
    private final AtomicLong totalFrees;
    private final AtomicLong poolHits;
    private final AtomicLong poolMisses;
    private final Arena arena;
    private final boolean isUnifiedMemory;

    /**
     * Create Metal memory pool.
     *
     * @param blockSize Size of each block in bytes
     * @param maxBlocks Maximum number of blocks
     * @param isUnifiedMemory True if running on Apple Silicon (unified memory)
     */
    public MetalMemoryPool(long blockSize, int maxBlocks, boolean isUnifiedMemory) {
        this.blockSize = blockSize;
        this.maxBlocks = maxBlocks;
        this.freeBlocks = new ConcurrentHashMap<>();
        this.allocatedBlocks = new ConcurrentHashMap<>();
        this.totalAllocations = new AtomicLong(0);
        this.totalFrees = new AtomicLong(0);
        this.poolHits = new AtomicLong(0);
        this.poolMisses = new AtomicLong(0);
        this.arena = Arena.ofShared();
        this.isUnifiedMemory = isUnifiedMemory;
        
        log.infof("Metal Memory Pool created: blockSize=%d bytes, maxBlocks=%d, unified=%s", 
                 blockSize, maxBlocks, isUnifiedMemory);
    }

    /**
     * Allocate memory from pool.
     *
     * @param size Size in bytes
     * @return Memory segment
     */
    public MemorySegment allocate(long size) {
        totalAllocations.incrementAndGet();
        
        // For unified memory, direct allocation is often faster
        if (isUnifiedMemory && size < blockSize / 2) {
            poolMisses.incrementAndGet();
            MemorySegment directAlloc = arena.allocate(size, 64);
            allocatedBlocks.put(directAlloc.address(), directAlloc);
            log.debugf("Direct allocation for unified memory: %d bytes", size);
            return directAlloc;
        }
        
        // Find block of appropriate size
        Long blockSizeKey = findBestFitBlock(size);
        
        if (blockSizeKey != null) {
            // Pool hit - reuse existing block
            MemorySegment block = freeBlocks.remove(blockSizeKey);
            if (block != null) {
                allocatedBlocks.put(block.address(), block);
                poolHits.incrementAndGet();
                log.debugf("Pool hit: allocated %d bytes from pool", size);
                return block;
            }
        }
        
        // Pool miss - allocate new
        poolMisses.incrementAndGet();
        MemorySegment newBlock = arena.allocate(size, 64);
        allocatedBlocks.put(newBlock.address(), newBlock);
        log.debugf("Pool miss: allocated %d bytes (new allocation)", size);
        return newBlock;
    }

    /**
     * Free memory back to pool.
     *
     * @param segment Memory segment to free
     */
    public void free(MemorySegment segment) {
        if (segment == null || segment.equals(MemorySegment.NULL)) {
            return;
        }
        
        totalFrees.incrementAndGet();
        long address = segment.address();
        
        MemorySegment removed = allocatedBlocks.remove(address);
        if (removed != null) {
            // Return to pool if not at max capacity
            if (freeBlocks.size() < maxBlocks) {
                freeBlocks.put(removed.byteSize(), removed);
                log.debugf("Freed %d bytes back to pool", removed.byteSize());
            } else {
                log.debugf("Pool full, deallocating %d bytes", removed.byteSize());
            }
        } else {
            log.debugf("Freed untracked segment at address %d", address);
        }
    }

    /**
     * Get pool statistics.
     */
    public PoolStats getStats() {
        return new PoolStats(
            blockSize,
            maxBlocks,
            freeBlocks.size(),
            allocatedBlocks.size(),
            totalAllocations.get(),
            totalFrees.get(),
            poolHits.get(),
            poolMisses.get(),
            calculateHitRate(),
            isUnifiedMemory
        );
    }

    /**
     * Clear all blocks from pool.
     */
    public void clear() {
        freeBlocks.clear();
        allocatedBlocks.clear();
        log.info("Metal Memory Pool cleared");
    }

    /**
     * Close the pool and release all resources.
     */
    public void close() {
        log.info("Closing Metal Memory Pool");
        clear();
        arena.close();
    }

    /**
     * Find best fit block for requested size.
     */
    private Long findBestFitBlock(long size) {
        return freeBlocks.keySet().stream()
            .filter(blockSize -> blockSize >= size)
            .min(Long::compare)
            .orElse(null);
    }

    /**
     * Calculate pool hit rate.
     */
    private double calculateHitRate() {
        long total = poolHits.get() + poolMisses.get();
        if (total == 0) return 0.0;
        return (double) poolHits.get() / total * 100.0;
    }

    /**
     * Pool statistics.
     */
    public static class PoolStats {
        public final long blockSize;
        public final int maxBlocks;
        public final int freeBlocks;
        public final int allocatedBlocks;
        public final long totalAllocations;
        public final long totalFrees;
        public final long poolHits;
        public final long poolMisses;
        public final double hitRate;
        public final boolean unifiedMemory;

        public PoolStats(long blockSize, int maxBlocks, int freeBlocks, 
                        int allocatedBlocks, long totalAllocations, long totalFrees,
                        long poolHits, long poolMisses, double hitRate, boolean unifiedMemory) {
            this.blockSize = blockSize;
            this.maxBlocks = maxBlocks;
            this.freeBlocks = freeBlocks;
            this.allocatedBlocks = allocatedBlocks;
            this.totalAllocations = totalAllocations;
            this.totalFrees = totalFrees;
            this.poolHits = poolHits;
            this.poolMisses = poolMisses;
            this.hitRate = hitRate;
            this.unifiedMemory = unifiedMemory;
        }

        @Override
        public String toString() {
            return String.format(
                "MetalPoolStats{blockSize=%d, free=%d, allocated=%d, hitRate=%.1f%%, allocations=%d, unified=%s}",
                blockSize, freeBlocks, allocatedBlocks, hitRate, totalAllocations, unifiedMemory
            );
        }
    }
}
