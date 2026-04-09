package tech.kayys.gollek.runtime.tensor;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Pool-aware tensor storage with atomic reference counting for the Gollek runtime.
 * <p>
 * This class wraps a native memory segment and manages its lifecycle through
 * reference counting. When the last reference is released, the underlying native
 * memory is returned to the {@link TensorPool} for reuse instead of being freed,
 * enabling zero-allocation tensor creation during graph execution.
 * <p>
 * <h2>Memory Management Strategy</h2>
 * <p>
 * Traditional tensor implementations allocate and free native memory for each
 * tensor operation, incurring significant malloc/free overhead. This class
 * implements a pooling strategy where:
 * </p>
 * <ul>
 *   <li>Memory is allocated once for a given shape/dtype/device combination</li>
 *   <li>Multiple tensor views can share the same storage via {@link #retain()}</li>
 *   <li>Memory is recycled to the pool when all references are released</li>
 *   <li>Subsequent tensors can reuse the pooled memory without allocation</li>
 * </ul>
 * <p>
 * This approach can yield 3×–10× performance improvements in real workloads by
 * eliminating allocation overhead and reducing GC pressure.
 * </p>
 * <p>
 * <h2>Reference Counting</h2>
 * <p>
 * Each {@code PooledTensorStorage} instance maintains an atomic reference count:
 * </p>
 * <ul>
 *   <li>Initial count is 1 upon construction</li>
 *   <li>{@link #retain()} increments the count (for creating views)</li>
 *   <li>{@link #release()} decrements the count</li>
 *   <li>When count reaches 0, memory is returned to the pool</li>
 * </ul>
 * <p>
 * <h2>Thread Safety</h2>
 * <p>
 * This class is thread-safe. Reference count operations use atomic CAS operations
 * to ensure correct behavior under concurrent access.
 * </p>
 * <p>
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Allocate storage through backend
 * MemorySegment handle = backend.allocate(size);
 * TensorKey key = new TensorKey(shape, dtype, device);
 * 
 * // Create pooled storage
 * PooledTensorStorage storage = new PooledTensorStorage(handle, key, pool, backend);
 * 
 * // Create multiple tensors sharing the same storage
 * Tensor tensor1 = new DefaultTensor(storage, shape, dtype, device);
 * storage.retain();  // Increment ref count for view
 * Tensor tensor2 = new DefaultTensor(storage, viewShape, dtype, device);
 * 
 * // When both tensors are closed, storage returns to pool
 * tensor1.close();  // refCount: 2 → 1
 * tensor2.close();  // refCount: 1 → 0, memory returned to pool
 * }</pre>
 *
 * @see TensorPool
 * @see TensorKey
 * @see DefaultTensor
 * @since 1.0
 */
public final class PooledTensorStorage {

    /**
     * The raw native memory segment containing tensor data.
     * This segment is owned by the TensorPool and must not be freed directly.
     */
    private final MemorySegment handle;

    /**
     * The key identifying this storage's memory layout (shape + dtype + device).
     * Used for pool indexing and reuse.
     */
    private final TensorKey key;

    /**
     * The pool that manages this storage's lifecycle.
     * May be null if storage was allocated outside the pool system.
     */
    private final TensorPool pool;

    /**
     * The backend type that allocated this memory.
     * Used for routing operations and proper cleanup.
     */
    private final BackendType backend;

    /**
     * Atomic reference count for memory management.
     * Starts at 1 and is incremented for each tensor view sharing this storage.
     * When it reaches 0, the memory is returned to the pool.
     */
    private final AtomicInteger refCount = new AtomicInteger(1);

    /**
     * Creates a new pooled tensor storage wrapper.
     *
     * @param handle the native memory segment containing tensor data
     * @param key the tensor key identifying shape, dtype, and device
     * @param pool the pool managing this storage (may be null)
     * @param backend the backend type that allocated the memory
     */
    public PooledTensorStorage(
        MemorySegment handle,
        TensorKey key,
        TensorPool pool,
        BackendType backend
    ) {
        this.handle = handle;
        this.key = key;
        this.pool = pool;
        this.backend = backend;
    }

    /**
     * Returns the raw native memory segment for this storage.
     * <p>
     * The returned segment points to the base address of the allocated memory.
     * Tensor views may access subsets of this memory using offset and stride.
     * </p>
     * <p>
     * <strong>Warning:</strong> Callers must not free this segment directly.
     * Memory management is handled automatically through {@link #release()}.
     * </p>
     *
     * @return the native memory segment
     */
    public MemorySegment handle() {
        return handle;
    }

    /**
     * Returns the backend type that owns this memory.
     * <p>
     * The backend type determines which native library can execute
     * operations on tensors using this storage.
     * </p>
     *
     * @return the backend type
     */
    public BackendType backend() {
        return backend;
    }

    /**
     * Increments the reference count for this storage.
     * <p>
     * Call this method when creating a new tensor view that shares
     * this storage. The storage will not be released until all views
     * are closed.
     * </p>
     * <p>
     * <strong>Thread Safety:</strong> This method is atomic and thread-safe.
     * </p>
     *
     * @see #release()
     */
    public void retain() {
        refCount.incrementAndGet();
    }

    /**
     * Decrements the reference count and potentially returns memory to the pool.
     * <p>
     * When the reference count reaches zero (i.e., all tensor views sharing
     * this storage have been closed), the underlying native memory is returned
     * to the {@link TensorPool} for reuse instead of being freed.
     * </p>
     * <p>
     * If no pool is configured (null), the memory will be reclaimed when the
     * enclosing {@link java.lang.foreign.Arena} closes.
     * </p>
     * <p>
     * <strong>Thread Safety:</strong> This method is atomic and thread-safe.
     * Multiple threads may safely call release concurrently.
     * </p>
     *
     * @see #retain()
     * @see TensorPool#release(TensorKey, MemorySegment)
     */
    public void release() {
        if (refCount.decrementAndGet() == 0) {
            if (pool != null) {
                pool.release(key, handle);
            }
            // If no pool, memory will be reclaimed when Arena closes
        }
    }

    /**
     * Returns the current reference count for diagnostic purposes.
     * <p>
     * This method is primarily useful for debugging and testing.
     * Production code should not rely on specific reference count values.
     * </p>
     *
     * @return the current number of active references to this storage
     */
    public int refCount() {
        return refCount.get();
    }
}
