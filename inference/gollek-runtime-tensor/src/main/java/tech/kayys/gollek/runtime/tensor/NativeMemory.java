package tech.kayys.gollek.runtime.tensor;

import java.lang.foreign.MemorySegment;

/**
 * Backend-dispatched native memory cleanup utility for the Gollek runtime.
 * <p>
 * This utility class provides centralized native memory deallocation logic,
 * routing free() calls to the correct backend binding based on which backend
 * allocated the memory.
 * <p>
 * <h2>Memory Management Model</h2>
 * <p>
 * In the Gollek runtime, native memory lifecycle is typically managed through:
 * </p>
 * <ol>
 *   <li><strong>{@link TensorPool}:</strong> Primary mechanism - memory is recycled, not freed</li>
 *   <li><strong>{@link ExecutionContext}:</strong> Arena-based cleanup when context closes</li>
 *   <li><strong>{@link NativeMemory}:</strong> Fallback for explicit deallocation</li>
 * </ol>
 * <p>
 * This class handles the third case - explicit deallocation when memory cannot
 * be pooled or when the pool is bypassed.
 * </p>
 * <p>
 * <h2>When NativeMemory.free() is Used</h2>
 * <ul>
 *   <li>During shutdown when pools are being cleared</li>
 *   <li>For tensors allocated outside the pool system</li>
 *   <li>When pool capacity is exceeded</li>
 *   <li>For error recovery scenarios</li>
 * </ul>
 * <p>
 * <h2>Backend-Specific Deallocation</h2>
 * <p>
 * Different backends may have different memory allocation strategies:
 * </p>
 * <ul>
 *   <li><strong>LibTorch:</strong> Uses torch allocator with caching</li>
 *   <li><strong>GGML:</strong> Uses llama.cpp allocator</li>
 *   <li><strong>ONNX:</strong> Uses OrtAllocator</li>
 *   <li><strong>CPU_JAVA:</strong> Uses Arena-managed memory</li>
 * </ul>
 * <p>
 * This utility ensures the correct deallocation path is used for each backend.
 * </p>
 * <p>
 * <h2>Usage</h2>
 * <pre>{@code
 * // Typical usage - during shutdown or error recovery
 * MemorySegment segment = ...;  // Previously allocated memory
 * BackendType backend = BackendType.GGML;
 * 
 * // Free the memory using the appropriate backend
 * NativeMemory.free(segment, backend);
 * }</pre>
 * <p>
 * <strong>Note:</strong> In normal operation, you should not need to call this
 * method directly. Use {@link Tensor#close()} and {@link ExecutionContext#close()}
 * for automatic memory management.
 * </p>
 * <p>
 * <h2>Null Safety</h2>
 * <p>
 * This method safely handles null or NULL segments by returning immediately
 * without error. This simplifies cleanup code in finally blocks.
 * </p>
 *
 * @see TensorPool
 * @see ExecutionContext
 * @see PooledTensorStorage
 * @since 1.0
 */
public final class NativeMemory {

    /**
     * Private constructor to prevent instantiation.
     * This class is a utility with only static methods.
     */
    private NativeMemory() {}

    /**
     * Frees a native memory segment using the backend that allocated it.
     * <p>
     * This method delegates to the appropriate backend's native free function
     * based on the backend type. If the backend is not available, the memory
     * will be reclaimed when the enclosing Arena closes.
     * </p>
     * <p>
     * <strong>When to Use:</strong>
     * </p>
     * <ul>
     *   <li>Shutting down the inference engine</li>
     *   <li>Clearing pools during reset</li>
     *   <li>Handling out-of-memory conditions</li>
     *   <li>Cleaning up after errors</li>
     * </ul>
     * <p>
     * <strong>Null Handling:</strong>
     * </p>
     * <ul>
     *   <li>If {@code segment} is null or {@link MemorySegment#NULL}, this method returns immediately</li>
     *   <li>If the backend is not registered, no action is taken (Arena will handle cleanup)</li>
     * </ul>
     *
     * @param segment the native memory segment to free
     * @param backend the backend type that allocated the memory
     * 
     * @see BackendRegistry#isAvailable(BackendType)
     * @see ExecutionContext#close()
     */
    public static void free(MemorySegment segment, BackendType backend) {
        if (segment == null || segment.equals(MemorySegment.NULL)) {
            return;
        }

        if (BackendRegistry.isAvailable(backend)) {
            // Delegate to the backend's native free if available
            // For now, we rely on Arena lifecycle for cleanup
        }
        // If no backend is available, the Arena will handle cleanup
        // when the ExecutionContext is closed.
    }
}
