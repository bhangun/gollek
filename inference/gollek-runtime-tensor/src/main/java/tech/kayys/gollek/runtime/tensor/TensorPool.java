package tech.kayys.gollek.runtime.tensor;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Thread-safe memory pool for reusing native tensor allocations in the Gollek runtime.
 * <p>
 * This class implements an object pool pattern for native memory segments, eliminating
 * the malloc/free overhead during tensor graph execution. Instead of freeing native
 * memory when a tensor is closed, the memory block is returned to this pool and can
 * be reused by future tensors with the same {@link TensorKey} (shape + dtype + device).
 * <p>
 * <h2>Performance Benefits</h2>
 * <p>
 * Native memory allocation (malloc/free) is expensive, especially in high-frequency
 * inference scenarios. By recycling memory blocks:
 * </p>
 * <ul>
 *   <li><strong>Reduced Latency:</strong> Pool acquisition is O(1) vs O(n) for malloc</li>
 *   <li><strong>Lower GC Pressure:</strong> Fewer Java object allocations</li>
 *   <li><strong>Better Cache Locality:</strong> Reusing same addresses improves CPU cache</li>
 *   <li><strong>Predictable Performance:</strong> No allocation spikes during inference</li>
 * </ul>
 * <p>
 * Real workloads can see 3×–10× speedup from memory pooling alone.
 * </p>
 * <p>
 * <h2>Pool Organization</h2>
 * <p>
 * Memory segments are organized in a two-level hierarchy:
 * </p>
 * <pre>
 * TensorPool
 *   ├─ TensorKey[shape=[2,3], dtype=FLOAT32, device=CUDA]
 *   │    ├─ MemorySegment[0x7f001]
 *   │    ├─ MemorySegment[0x7f002]
 *   │    └─ MemorySegment[0x7f003]
 *   └─ TensorKey[shape=[4,5], dtype=FLOAT16, device=CUDA]
 *        ├─ MemorySegment[0x7f004]
 *        └─ MemorySegment[0x7f005]
 * </pre>
 * <p>
 * Each unique TensorKey maintains a queue of available memory segments. When a
 * tensor is closed, its memory is enqueued. When a new tensor needs memory, it
 * first checks the pool before allocating fresh memory.
 * </p>
 * <p>
 * <h2>Thread Safety</h2>
 * <p>
 * This class is fully thread-safe:
 * </p>
 * <ul>
 *   <li>Uses {@link ConcurrentHashMap} for thread-safe key indexing</li>
 *   <li>Uses {@link ConcurrentLinkedQueue} for lock-free segment queues</li>
 *   <li>Uses {@link AtomicLong} for lock-free statistics counters</li>
 * </ul>
 * <p>
 * Multiple threads can safely acquire and release memory concurrently without
 * external synchronization.
 * </p>
 * <p>
 * <h2>Usage Pattern</h2>
 * <pre>{@code
 * // Create shared pool (typically singleton per backend)
 * TensorPool pool = new TensorPool();
 * 
 * // Acquire memory for a tensor
 * TensorKey key = new TensorKey(new long[]{2, 3}, DType.FLOAT32, Device.CUDA);
 * MemorySegment segment = pool.acquire(key, backend, ctx);
 * 
 * // Use segment for tensor...
 * PooledTensorStorage storage = new PooledTensorStorage(segment, key, pool, backend);
 * 
 * // When tensor is closed, memory returns to pool automatically
 * tensor.close();  // Calls pool.release(key, segment)
 * 
 * // Monitor pool efficiency
 * System.out.println("Hit rate: " + pool.hitRate() * 100 + "%");
 * }</pre>
 * <p>
 * <h2>Pool Statistics</h2>
 * <p>
 * The pool tracks hit/miss statistics for performance monitoring:
 * </p>
 * <ul>
 *   <li><strong>Hits:</strong> Number of successful pool reuses</li>
 *   <li><strong>Misses:</strong> Number of fresh allocations</li>
 *   <li><strong>Hit Rate:</strong> Percentage of allocations served from pool</li>
 * </ul>
 * <p>
 * A high hit rate (>80%) indicates effective pooling. Low hit rates may suggest:
 * </p>
 * <ul>
 *   <li>Too many unique tensor shapes (consider shape optimization)</li>
 *   <li>Pool cleared too frequently</li>
 *   <li>Mismatched tensor lifecycles</li>
 * </ul>
 *
 * @see TensorKey
 * @see PooledTensorStorage
 * @see ExecutionContext
 * @since 1.0
 */
public final class TensorPool {

    /**
     * Map from TensorKey to queue of pooled memory segments.
     * Each key maintains a FIFO queue of available segments for reuse.
     */
    private final ConcurrentMap<TensorKey, ConcurrentLinkedQueue<MemorySegment>> pool =
        new ConcurrentHashMap<>();

    /**
     * Counter for successful pool hits (reused allocations).
     * Used for monitoring pool efficiency.
     */
    private final AtomicLong hits = new AtomicLong(0);

    /**
     * Counter for pool misses (fresh allocations required).
     * High miss rate indicates poor pool utilization.
     */
    private final AtomicLong misses = new AtomicLong(0);

    /**
     * Acquires a native memory segment for the given tensor key.
     * <p>
     * This method first checks the pool for an available segment with matching
     * shape, dtype, and device. If found, the segment is dequeued and returned.
     * Otherwise, a fresh allocation is performed via the backend.
     * </p>
     * <p>
     * <strong>Acquisition Strategy:</strong>
     * </p>
     * <ol>
     *   <li>Look up queue for the given TensorKey</li>
     *   <li>If queue exists and non-empty, poll a segment (hit)</li>
     *   <li>If queue empty or missing, allocate fresh memory via backend (miss)</li>
     * </ol>
     *
     * @param key     tensor dimensions, dtype, and device
     * @param backend backend to allocate from on cache miss
     * @param ctx     execution context for allocation scope
     * @return a native memory segment, either pooled or freshly allocated
     * @throws NullPointerException if key, backend, or ctx is null
     */
    public MemorySegment acquire(TensorKey key, Backend backend, ExecutionContext ctx) {
        var queue = pool.get(key);
        if (queue != null) {
            MemorySegment seg = queue.poll();
            if (seg != null) {
                hits.incrementAndGet();
                return seg;
            }
        }

        misses.incrementAndGet();
        return backend.createTensor(key.shape(), key.dtype(), key.device(), ctx)
            .nativeHandle();
    }

    /**
     * Returns a memory segment to the pool for future reuse.
     * <p>
     * This method is called automatically when a {@link PooledTensorStorage} is
     * released and its reference count reaches zero. The segment is enqueued
     * under its TensorKey for reuse by future tensors with matching layout.
     * </p>
     * <p>
     * <strong>Note:</strong> The segment is not freed here. It remains valid
     * and will be reused by the next {@link #acquire(TensorKey, Backend, ExecutionContext)}
     * call with a matching key.
     * </p>
     *
     * @param key     tensor key identifying the memory layout
     * @param segment native memory segment to recycle
     * @throws NullPointerException if key or segment is null
     */
    public void release(TensorKey key, MemorySegment segment) {
        pool.computeIfAbsent(key, k -> new ConcurrentLinkedQueue<>())
            .offer(segment);
    }

    /**
     * Returns the number of successful pool hits (reused allocations).
     * <p>
     * A high hit count indicates effective memory reuse and reduced allocation
     * overhead. Compare with {@link #misses()} to calculate hit rate.
     * </p>
     *
     * @return total number of pool hits since creation
     * @see #hitRate()
     */
    public long hits() {
        return hits.get();
    }

    /**
     * Returns the number of pool misses (fresh allocations).
     * <p>
     * Misses occur when the pool cannot satisfy an allocation request, requiring
     * a fresh malloc via the backend. High miss rates may indicate:
     * </p>
     * <ul>
     *   <li>Diverse tensor shapes not seen before</li>
     *   <li>Pool was cleared or segments not returned</li>
     *   <li>First-time allocations during warmup</li>
     * </ul>
     *
     * @return total number of pool misses since creation
     * @see #hits()
     */
    public long misses() {
        return misses.get();
    }

    /**
     * Returns the pool hit rate as a ratio between 0.0 and 1.0.
     * <p>
     * Hit rate is calculated as: {@code hits / (hits + misses)}
     * </p>
     * <p>
     * <strong>Interpretation:</strong>
     * </p>
     * <ul>
     *   <li>0.0 = All allocations were fresh (no reuse)</li>
     *   <li>0.5 = Half of allocations were reused</li>
     *   <li>1.0 = All allocations served from pool (ideal)</li>
     * </ul>
     * <p>
     * Production workloads should target >0.8 hit rate for optimal performance.
     * </p>
     *
     * @return hit rate ratio (0.0 to 1.0), or 0.0 if no allocations yet
     */
    public double hitRate() {
        long total = hits.get() + misses.get();
        return total == 0 ? 0.0 : (double) hits.get() / total;
    }

    /**
     * Returns the total number of memory segments currently pooled across all keys.
     * <p>
     * This count represents idle memory that is available for reuse but not
     * currently in use by any tensor. High pooled counts during idle periods
     * are normal; high counts during active inference may indicate over-provisioning.
     * </p>
     *
     * @return total number of segments available in the pool
     */
    public long pooledCount() {
        return pool.values().stream().mapToLong(ConcurrentLinkedQueue::size).sum();
    }

    /**
     * Clears all pooled segments from the pool.
     * <p>
     * <strong>Warning:</strong> This method does NOT free the native memory.
     * Cleared segments will be garbage collected only when their owning Arena
     * closes or when explicitly freed via {@link NativeMemory#free(MemorySegment, BackendType)}.
     * </p>
     * <p>
     * Use this method cautiously, typically during shutdown or when resetting
     * the inference engine state.
     * </p>
     */
    public void clear() {
        pool.clear();
    }
}
