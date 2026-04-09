package tech.kayys.gollek.runtime.tensor;

import java.lang.foreign.Arena;
import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Scoped execution context for managing temporary tensor lifecycles during inference.
 * <p>
 * This class provides automatic memory management for tensor operations within a
 * single inference pass. Instead of each operation allocating and freeing its own
 * {@link Arena}, all temporary tensors share a single context that is cleaned up
 * when the context closes.
 * <p>
 * <h2>Memory Management Model</h2>
 * <p>
 * The execution context implements a scoped memory management pattern:
 * </p>
 * <ul>
 *   <li><strong>Creation:</strong> A new context creates a confined Arena</li>
 *   <li><strong>Tracking:</strong> Temporary tensors register with the context</li>
 *   <li><strong>Cleanup:</strong> When closed, all tracked tensors are released in LIFO order</li>
 *   <li><strong>Arena Close:</strong> The underlying Arena is closed, freeing all native memory</li>
 * </ul>
 * <p>
 * This approach eliminates manual memory management and prevents memory leaks in
 * complex inference graphs with many intermediate tensors.
 * </p>
 * <p>
 * <h2>Usage Pattern</h2>
 * <pre>{@code
 * try (var ctx = new ExecutionContext()) {
 *     // Create tensors through backend - automatically tracked
 *     Tensor a = backend.createTensor(shape1, dtype, device, ctx);
 *     Tensor b = backend.createTensor(shape2, dtype, device, ctx);
 *     
 *     // Chain operations - intermediates are tracked
 *     Tensor result = a.matmul(b, ctx).relu(ctx);
 *     
 *     // Use result...
 *     process(result);
 *     
 *     // All tensors automatically cleaned up here:
 *     // 1. Tracked tensors closed in reverse order (LIFO)
 *     // 2. Arena closed, freeing all native memory
 * }
 * }</pre>
 * <p>
 * <h2>Thread Safety</h2>
 * <p>
 * This class is not thread-safe. Each thread should use its own execution context.
 * For concurrent inference, create separate contexts per thread or per request.
 * </p>
 * <p>
 * <h2>Arena Confinement</h2>
 * <p>
 * The context uses a {@link Arena#ofConfined() confined Arena} which provides:
 * </p>
 * <ul>
 *   <li><strong>Thread confinement:</strong> Arena can only be accessed by owning thread</li>
 *   <li><strong>Explicit lifecycle:</strong> Memory freed only when Arena is closed</li>
 *   <li><strong>Zero overhead:</strong> No reference counting or GC pressure</li>
 * </ul>
 *
 * @see Arena
 * @see Tensor
 * @see Backend
 * @since 1.0
 */
public final class ExecutionContext implements AutoCloseable {

    /**
     * The shared Arena for all native memory allocations in this execution scope.
     * All tensors created within this context share the same Arena.
     */
    private final Arena arena;

    /**
     * Stack of temporary tensors to be cleaned up when this context closes.
     * Tensors are released in LIFO (Last-In-First-Out) order to ensure
     * dependent tensors are released before their dependencies.
     */
    private final Deque<Tensor> temps = new ArrayDeque<>();

    /**
     * Creates a new execution context with a confined Arena.
     * <p>
     * The confined Arena ensures thread-safe memory management within the
     * creating thread's scope.
     * </p>
     */
    public ExecutionContext() {
        this.arena = Arena.ofConfined();
    }

    /**
     * Returns the shared Arena for this execution scope.
     * <p>
     * Backends use this Arena to allocate native memory segments for tensors.
     * All allocations within the same Arena share the same lifecycle.
     * </p>
     * <p>
     * <strong>Warning:</strong> Callers should not close the Arena directly.
     * Use {@link #close()} to ensure proper cleanup of tracked tensors.
     * </p>
     *
     * @return the shared Arena
     */
    public Arena arena() {
        return arena;
    }

    /**
     * Tracks a temporary tensor for automatic cleanup when this context closes.
     * <p>
     * Call this method for tensors that should be automatically released when
     * the execution scope ends. The tensor is added to a LIFO stack and will
     * be closed in reverse order of registration.
     * </p>
     * <p>
     * This method supports fluent chaining:
     * </p>
     * <pre>{@code
     * Tensor result = ctx.track(backend.createTensor(...));
     * }</pre>
     *
     * @param tensor the tensor to track for cleanup
     * @param <T>    the tensor subtype
     * @return the same tensor instance for fluent chaining
     * @throws NullPointerException if tensor is null
     */
    public <T extends Tensor> T track(T tensor) {
        temps.push(tensor);
        return tensor;
    }

    /**
     * Closes this execution context, releasing all tracked tensors and native memory.
     * <p>
     * Cleanup occurs in two phases:
     * </p>
     * <ol>
     *   <li><strong>Tensor Cleanup:</strong> All tracked tensors are closed in LIFO order.
     *       This allows tensors to return their memory to pools before the Arena closes.</li>
     *   <li><strong>Arena Close:</strong> The underlying Arena is closed, freeing any
     *       remaining native memory not returned to pools.</li>
     * </ol>
     * <p>
     * Exceptions during individual tensor cleanup are caught and ignored to ensure
     * complete cleanup of all resources.
     * </p>
     * <p>
     * After closing, this context should not be reused. Create a new context for
     * subsequent inference operations.
     * </p>
     */
    @Override
    public void close() {
        // Release tracked tensors in reverse order (LIFO)
        while (!temps.isEmpty()) {
            try {
                temps.pop().close();
            } catch (Exception ignored) {
                // Best-effort cleanup - continue with remaining tensors
            }
        }
        try {
            arena.close();
        } catch (Exception ignored) {
            // Arena close should not fail in normal operation
        }
    }
}
