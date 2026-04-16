package tech.kayys.gollek.multimodal.gpu;

import org.jboss.logging.Logger;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * GPU memory pool for efficient memory management.
 * Reduces allocation overhead by reusing memory blocks.
 */
public class GPUMemoryPool {

    private static final Logger log = Logger.getLogger(GPUMemoryPool.class);

    private final long blockSize;
    private final int maxBlocks;
    private final ConcurrentHashMap<Long, MemorySegment> freeBlocks;
    private final ConcurrentHashMap<Long, MemorySegment> allocatedBlocks;
    private final AtomicLong totalAllocations;
    private final AtomicLong totalFrees;
    private final AtomicLong poolHits;
    private final AtomicLong poolMisses;
    private final Arena arena;

    /**
     * Create GPU memory pool.
     *
     * @param blockSize Size of each block in bytes
     * @param maxBlocks Maximum number of blocks
     */
    public GPUMemoryPool(long blockSize, int maxBlocks) {
        this.blockSize = blockSize;
        this.maxBlocks = maxBlocks;
        this.freeBlocks = new ConcurrentHashMap<>();
        this.allocatedBlocks = new ConcurrentHashMap<>();
        this.totalAllocations = new AtomicLong(0);
        this.totalFrees = new AtomicLong(0);
        this.poolHits = new AtomicLong(0);
        this.poolMisses = new AtomicLong(0);
        this.arena = Arena.ofShared();
        
        log.infof("GPU Memory Pool created: blockSize=%d bytes, maxBlocks=%d", 
                 blockSize, maxBlocks);
    }

    /**
     * Allocate memory from pool.
     *
     * @param size Size in bytes
     * @return Memory segment
     */
    public MemorySegment allocate(long size) {
        totalAllocations.incrementAndGet();
        
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
                // Don't explicitly free - let arena handle it
            }
        } else {
            log.warnf("Attempted to free untracked segment at address %d", address);
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
            calculateHitRate()
        );
    }

    /**
     * Clear all blocks from pool.
     */
    public void clear() {
        freeBlocks.clear();
        allocatedBlocks.clear();
        log.info("GPU Memory Pool cleared");
    }

    /**
     * Close the pool and release all resources.
     */
    public void close() {
        log.info("Closing GPU Memory Pool");
        clear();
        arena.close();
    }

    /**
     * Find best fit block for requested size.
     */
    private Long findBestFitBlock(long size) {
        // Find smallest block that fits
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

        public PoolStats(long blockSize, int maxBlocks, int freeBlocks, 
                        int allocatedBlocks, long totalAllocations, long totalFrees,
                        long poolHits, long poolMisses, double hitRate) {
            this.blockSize = blockSize;
            this.maxBlocks = maxBlocks;
            this.freeBlocks = freeBlocks;
            this.allocatedBlocks = allocatedBlocks;
            this.totalAllocations = totalAllocations;
            this.totalFrees = totalFrees;
            this.poolHits = poolHits;
            this.poolMisses = poolMisses;
            this.hitRate = hitRate;
        }

        @Override
        public String toString() {
            return String.format(
                "PoolStats{blockSize=%d, free=%d, allocated=%d, hitRate=%.1f%%, allocations=%d, frees=%d}",
                blockSize, freeBlocks, allocatedBlocks, hitRate, totalAllocations, totalFrees
            );
        }
    }
}
