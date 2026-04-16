/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * NativeMemoryPool.java
 * ───────────────────────
 * Native memory pool for efficient buffer reuse.
 */
package tech.kayys.gollek.safetensor.audio.processing;

import org.jboss.logging.Logger;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * High-performance native memory pool for FFM buffer reuse.
 * 
 * <p>This pool manages native memory segments to avoid repeated allocation/deallocation
 * overhead during audio decoding. Buffers are reused across decode operations,
 * significantly reducing GC pressure and improving throughput.</p>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * try (NativeMemoryPool pool = NativeMemoryPool.create(1024 * 1024, 10)) {
 *     try (PooledBuffer buffer = pool.acquire()) {
 *         MemorySegment segment = buffer.segment();
 *         // Use native memory...
 *     } // Buffer automatically returned to pool
 * } // Pool closed, all buffers freed
 * }</pre>
 *
 * <h2>Features</h2>
 * <ul>
 *   <li>Fixed-size buffer pools to prevent fragmentation</li>
 *   <li>Automatic buffer return on close (try-with-resources)</li>
 *   <li>Configurable pool size and buffer capacity</li>
 *   <li>Thread-safe acquisition and release</li>
 *   <li>Statistics tracking for monitoring</li>
 *   <li>Overflow handling with temporary buffers</li>
 * </ul>
 *
 * @author Bhangun
 * @version 1.0.0
 * @since 2.0.0
 */
public final class NativeMemoryPool implements AutoCloseable {

    private static final Logger log = Logger.getLogger(NativeMemoryPool.class);

    /**
     * Default buffer capacity: 1MB (sufficient for most audio files).
     */
    public static final long DEFAULT_BUFFER_CAPACITY = 1024 * 1024;

    /**
     * Default pool size: 10 buffers.
     */
    public static final int DEFAULT_POOL_SIZE = 10;

    /**
     * Buffer capacity in bytes.
     */
    private final long bufferCapacity;

    /**
     * Maximum number of pooled buffers.
     */
    private final int poolSize;

    /**
     * Queue of available buffers.
     */
    private final BlockingQueue<PooledBuffer> availableBuffers;

    /**
     * Global arena for buffer lifetime management.
     */
    private final Arena arena;

    /**
     * Pool statistics.
     */
    private final java.util.concurrent.atomic.LongAdder acquireCount = new java.util.concurrent.atomic.LongAdder();
    private final java.util.concurrent.atomic.LongAdder hitCount = new java.util.concurrent.atomic.LongAdder();
    private final java.util.concurrent.atomic.LongAdder missCount = new java.util.concurrent.atomic.LongAdder();
    private final java.util.concurrent.atomic.LongAdder timeoutCount = new java.util.concurrent.atomic.LongAdder();

    /**
     * Pool state.
     */
    private final AtomicBoolean closed = new AtomicBoolean(false);

    /**
     * Count of created buffers.
     */
    private final AtomicInteger createdCount = new AtomicInteger(0);

    /**
     * Create a new memory pool with default settings.
     *
     * @return new memory pool
     */
    public static NativeMemoryPool create() {
        return create(DEFAULT_BUFFER_CAPACITY, DEFAULT_POOL_SIZE);
    }

    /**
     * Create a new memory pool with specified settings.
     *
     * @param bufferCapacity capacity of each buffer in bytes
     * @param poolSize       maximum number of buffers to pool
     * @return new memory pool
     */
    public static NativeMemoryPool create(long bufferCapacity, int poolSize) {
        return new NativeMemoryPool(bufferCapacity, poolSize);
    }

    /**
     * Create a new memory pool.
     *
     * @param bufferCapacity capacity of each buffer in bytes
     * @param poolSize       maximum number of buffers to pool
     */
    private NativeMemoryPool(long bufferCapacity, int poolSize) {
        this.bufferCapacity = bufferCapacity;
        this.poolSize = poolSize;
        this.arena = Arena.ofShared();
        this.availableBuffers = new ArrayBlockingQueue<>(poolSize);

        // Pre-allocate buffers
        for (int i = 0; i < poolSize; i++) {
            try {
                MemorySegment segment = arena.allocate(bufferCapacity);
                PooledBuffer buffer = new PooledBuffer(segment, i);
                availableBuffers.offer(buffer);
                createdCount.incrementAndGet();
            } catch (OutOfMemoryError e) {
                log.warnf("Failed to pre-allocate buffer %d: %s", i, e.getMessage());
                break;
            }
        }

        log.debugf("NativeMemoryPool created: capacity=%d bytes, poolSize=%d, allocated=%d",
                bufferCapacity, poolSize, createdCount.get());
    }

    /**
     * Acquire a buffer from the pool.
     *
     * @return pooled buffer (must be closed after use)
     * @throws IllegalStateException if pool is closed
     */
    public PooledBuffer acquire() {
        if (closed.get()) {
            throw new IllegalStateException("Memory pool is closed");
        }

        PooledBuffer buffer = availableBuffers.poll();
        acquireCount.add(1);

        if (buffer != null) {
            hitCount.add(1);
            buffer.reset();
            return buffer;
        }

        // Pool exhausted - create temporary buffer
        missCount.add(1);
        log.trace("Pool exhausted, creating temporary buffer");
        MemorySegment tempSegment = arena.allocate(bufferCapacity);
        return new PooledBuffer(tempSegment, -1, true);
    }

    /**
     * Acquire a buffer with timeout.
     *
     * @param timeout maximum time to wait
     * @param unit    time unit
     * @return pooled buffer, or null if timeout elapsed
     * @throws InterruptedException if interrupted while waiting
     */
    public PooledBuffer acquire(long timeout, java.util.concurrent.TimeUnit unit)
            throws InterruptedException {
        if (closed.get()) {
            throw new IllegalStateException("Memory pool is closed");
        }

        PooledBuffer buffer = availableBuffers.poll(timeout, unit);
        acquireCount.add(1);

        if (buffer != null) {
            hitCount.add(1);
            buffer.reset();
            return buffer;
        }

        missCount.add(1);
        timeoutCount.add(1);
        return null;
    }

    /**
     * Return a buffer to the pool.
     *
     * @param buffer buffer to return
     */
    void release(PooledBuffer buffer) {
        if (buffer.isTemporary || closed.get()) {
            // Don't return temporary buffers or release to closed pool
            return;
        }

        buffer.clear();
        if (!availableBuffers.offer(buffer)) {
            log.warnf("Failed to return buffer %d to full pool", buffer.id);
        }
    }

    /**
     * Get pool statistics.
     *
     * @return current statistics
     */
    public PoolStats getStats() {
        return new PoolStats(
            acquireCount.sum(),
            hitCount.sum(),
            missCount.sum(),
            timeoutCount.sum()
        );
    }

    /**
     * Get the number of available buffers.
     *
     * @return available buffer count
     */
    public int getAvailableCount() {
        return availableBuffers.size();
    }

    /**
     * Get the total number of created buffers.
     *
     * @return created buffer count
     */
    public int getCreatedCount() {
        return createdCount.get();
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            log.debugf("Closing NativeMemoryPool: %d buffers, stats=%s",
                    createdCount.get(), getStats());
            arena.close();
        }
    }

    /**
     * Pooled native memory buffer.
     * 
     * <p>Implements {@link AutoCloseable} for automatic return to pool
     * when used in try-with-resources statements.</p>
     */
    public final class PooledBuffer implements AutoCloseable {

        /**
         * Underlying memory segment.
         */
        private final MemorySegment segment;

        /**
         * Buffer ID (-1 for temporary buffers).
         */
        private final int id;

        /**
         * Whether this is a temporary (non-pooled) buffer.
         */
        private final boolean isTemporary;

        /**
         * Whether this buffer has been returned to pool.
         */
        private final AtomicBoolean returned = new AtomicBoolean(false);

        /**
         * Bytes written to this buffer.
         */
        private long bytesWritten = 0;

        /**
         * Create a pooled buffer.
         *
         * @param segment     underlying memory segment
         * @param id          buffer ID
         * @param isTemporary whether this is a temporary buffer
         */
        private PooledBuffer(MemorySegment segment, int id, boolean isTemporary) {
            this.segment = segment;
            this.id = id;
            this.isTemporary = isTemporary;
        }

        /**
         * Create a pooled buffer (non-temporary).
         *
         * @param segment underlying memory segment
         * @param id      buffer ID
         */
        private PooledBuffer(MemorySegment segment, int id) {
            this(segment, id, false);
        }

        /**
         * Get the underlying memory segment.
         *
         * @return memory segment
         */
        public MemorySegment segment() {
            return segment;
        }

        /**
         * Get the buffer capacity in bytes.
         *
         * @return capacity
         */
        public long capacity() {
            return bufferCapacity;
        }

        /**
         * Get the buffer ID.
         *
         * @return buffer ID (-1 for temporary buffers)
         */
        public int id() {
            return id;
        }

        /**
         * Check if this is a temporary buffer.
         *
         * @return true if temporary
         */
        public boolean isTemporary() {
            return isTemporary;
        }

        /**
         * Get bytes written to this buffer.
         *
         * @return bytes written
         */
        public long bytesWritten() {
            return bytesWritten;
        }

        /**
         * Set bytes written (for tracking).
         *
         * @param bytesWritten bytes written
         */
        public void setBytesWritten(long bytesWritten) {
            this.bytesWritten = bytesWritten;
        }

        /**
         * Clear the buffer (zero memory, reset position).
         */
        void clear() {
            segment.fill((byte) 0);
            bytesWritten = 0;
        }

        /**
         * Reset buffer state for reuse.
         */
        void reset() {
            bytesWritten = 0;
            returned.set(false);
        }

        /**
         * Create a slice of this buffer.
         *
         * @param offset offset in bytes
         * @param size   size in bytes
         * @return sliced memory segment
         */
        public MemorySegment slice(long offset, long size) {
            return segment.asSlice(offset, size);
        }

        /**
         * Copy data to this buffer.
         *
         * @param src  source bytes
         * @param size number of bytes to copy
         */
        public void copyFrom(byte[] src, int size) {
            MemorySegment.copy(MemorySegment.ofArray(src), 0, segment, 0, size);
            bytesWritten = size;
        }

        /**
         * Copy data from this buffer.
         *
         * @param dest destination bytes
         * @param size number of bytes to copy
         */
        public void copyTo(byte[] dest, int size) {
            MemorySegment.copy(segment, 0, MemorySegment.ofArray(dest), 0, size);
        }

        @Override
        public void close() {
            if (returned.compareAndSet(false, true)) {
                NativeMemoryPool.this.release(this);
            }
        }

        @Override
        public String toString() {
            return String.format("PooledBuffer[id=%d, capacity=%d, temporary=%s]",
                    id, bufferCapacity, isTemporary);
        }
    }

    /**
     * Pool statistics snapshot.
     *
     * @param acquireCount total acquire operations
     * @param hitCount     successful pool hits
     * @param missCount    pool misses (temporary buffers created)
     * @param timeoutCount acquire timeouts
     */
    public record PoolStats(
            long acquireCount,
            long hitCount,
            long missCount,
            long timeoutCount
    ) {
        /**
         * Calculate hit rate.
         *
         * @return hit rate as percentage (0-100)
         */
        public double hitRate() {
            return acquireCount > 0 ? (hitCount * 100.0 / acquireCount) : 0.0;
        }

        @Override
        public String toString() {
            return String.format("PoolStats[acquires=%d, hits=%d, misses=%d, hitRate=%.1f%%]",
                    acquireCount, hitCount, missCount, hitRate());
        }
    }
}
