/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

import jakarta.enterprise.context.ApplicationScoped;

/**
 * Global pool for off-heap MemorySegments.
 * Minimizes allocation overhead and fragmentation for the GKV memory fabric.
 */
@ApplicationScoped
public class KVPool {

    private final Map<Long, ConcurrentLinkedQueue<MemorySegment>> segmentsBySize = new ConcurrentHashMap<>();
    private final Arena globalArena = Arena.ofAuto();

    /**
     * Acquires a memory segment of at least the requested size.
     * Aligns to 64 bytes.
     */
    public MemorySegment acquire(long size) {
        long alignedSize = align(size, 64);
        ConcurrentLinkedQueue<MemorySegment> queue = segmentsBySize.computeIfAbsent(alignedSize, k -> new ConcurrentLinkedQueue<>());
        
        MemorySegment segment = queue.poll();
        if (segment == null) {
            segment = globalArena.allocate(alignedSize, 64);
        }
        
        return segment;
    }

    /**
     * Returns a segment to the pool for reuse.
     */
    public void release(MemorySegment segment) {
        long size = segment.byteSize();
        ConcurrentLinkedQueue<MemorySegment> queue = segmentsBySize.get(size);
        if (queue != null) {
            queue.offer(segment);
        }
    }

    /**
     * Closes the global arena and clears all pooled memory.
     */
    public void shutdown() {
        globalArena.close();
        segmentsBySize.clear();
    }

    private static long align(long offset, int alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }
}
