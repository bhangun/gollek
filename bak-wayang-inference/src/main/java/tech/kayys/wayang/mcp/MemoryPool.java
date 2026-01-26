package tech.kayys.golek.mcp;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MemoryPool {
    private static final Logger log = LoggerFactory.getLogger(MemoryPool.class);
    
    private final ConcurrentLinkedQueue<MemorySegment> pool;
    private final long segmentSize;
    private final int maxPoolSize;
    private final AtomicLong allocatedCount = new AtomicLong(0);
    private final AtomicLong poolHits = new AtomicLong(0);
    private final AtomicLong poolMisses = new AtomicLong(0);
    private final Arena arena;
    
    public MemoryPool(long segmentSize, int maxPoolSize) {
        this.segmentSize = segmentSize;
        this.maxPoolSize = maxPoolSize;
        this.pool = new ConcurrentLinkedQueue<>();
        this.arena = Arena.ofShared();
        
        // Pre-allocate some segments
        for (int i = 0; i < Math.min(10, maxPoolSize); i++) {
            pool.offer(arena.allocate(segmentSize));
        }
    }
    
    public MemorySegment acquire() {
        MemorySegment segment = pool.poll();
        
        if (segment != null) {
            poolHits.incrementAndGet();
            return segment;
        }
        
        poolMisses.incrementAndGet();
        allocatedCount.incrementAndGet();
        return arena.allocate(segmentSize);
    }
    
    public void release(MemorySegment segment) {
        if (pool.size() < maxPoolSize) {
            pool.offer(segment);
        }
    }
    
    public MemoryPoolStats getStats() {
        return new MemoryPoolStats(
            allocatedCount.get(),
            pool.size(),
            poolHits.get(),
            poolMisses.get(),
            getHitRate()
        );
    }
    
    private double getHitRate() {
        long hits = poolHits.get();
        long misses = poolMisses.get();
        long total = hits + misses;
        return total > 0 ? (double) hits / total : 0.0;
    }
    
    public void clear() {
        pool.clear();
    }
    
    public void close() {
        clear();
        arena.close();
    }
    
    public record MemoryPoolStats(
        long allocated,
        int pooled,
        long hits,
        long misses,
        double hitRate
    ) {}
}
