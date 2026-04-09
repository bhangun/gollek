package tech.kayys.gollek.runtime.tensor;

import java.util.Arrays;

/**
 * Pool index key for native memory reuse in the Gollek inference runtime.
 * <p>
 * This record serves as a unique identifier for tensor memory layouts, enabling
 * efficient memory pooling and reuse. Two tensors with the same key (shape + dtype
 * + device) can share the same underlying native memory block, eliminating
 * allocation overhead during graph execution.
 * <p>
 * <h2>Key Components</h2>
 * <ul>
 *   <li><strong>shape:</strong> The tensor dimensions (e.g., [2, 3, 4])</li>
 *   <li><strong>dtype:</strong> The data type (e.g., FLOAT32, INT8)</li>
 *   <li><strong>device:</strong> The target compute device (e.g., CPU, CUDA)</li>
 * </ul>
 * <p>
 * Together, these three components uniquely identify a memory layout. Tensors
 * differing in any component require separate memory allocations.
 * </p>
 * <p>
 * <h2>Equality and Hashing</h2>
 * <p>
 * This record implements value-based equality:
 * </p>
 * <ul>
 *   <li>Two keys are equal if all three components are equal</li>
 *   <li>Shape arrays are compared using {@link Arrays#equals(long[], long[])}</li>
 *   <li>Hash code combines all components for efficient HashMap lookup</li>
 * </ul>
 * <p>
 * <h2>Usage in TensorPool</h2>
 * <pre>{@code
 * // Create a key for a specific tensor layout
 * TensorKey key = new TensorKey(new long[]{2, 3}, DType.FLOAT32, Device.CUDA);
 * 
 * // Acquire memory from pool using the key
 * MemorySegment segment = pool.acquire(key, backend, ctx);
 * 
 * // Later, same key retrieves pooled memory
 * MemorySegment reused = pool.acquire(key, backend, ctx);  // Likely a hit!
 * }</pre>
 * <p>
 * <h2>Memory Efficiency</h2>
 * <p>
 * The defensive copy of the shape array ensures that external modifications
 * don't corrupt pool indexing. This is critical for thread-safe pool operation.
 * </p>
 *
 * @param shape  the tensor shape (defensively copied)
 * @param dtype  the tensor data type
 * @param device the target compute device
 * @see TensorPool
 * @see PooledTensorStorage
 * @since 1.0
 */
public record TensorKey(long[] shape, DType dtype, Device device) {

    /**
     * Compact constructor that performs defensive copy of the shape array.
     * <p>
     * This ensures that external modifications to the shape array don't affect
     * pool indexing or equality checks. The copy is transparent to callers.
     * </p>
     */
    public TensorKey {
        shape = shape.clone();
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof TensorKey k)) return false;
        return Arrays.equals(shape, k.shape)
            && dtype == k.dtype
            && device == k.device;
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(shape) * 31 * 31
            + dtype.hashCode() * 31
            + device.hashCode();
    }

    @Override
    public String toString() {
        return "TensorKey[shape=" + Arrays.toString(shape)
            + ", dtype=" + dtype
            + ", device=" + device + "]";
    }
}
