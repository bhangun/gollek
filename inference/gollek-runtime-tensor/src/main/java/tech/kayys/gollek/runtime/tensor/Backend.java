package tech.kayys.gollek.runtime.tensor;

/**
 * Pluggable execution backend for tensor operations in the Gollek inference runtime.
 * <p>
 * This interface defines the contract for native inference backends that provide
 * hardware-accelerated tensor computations. Each backend (LibTorch, GGML, ONNX Runtime,
 * LiteRT) implements this interface to expose its native capabilities through a
 * unified API.
 * <p>
 * <h2>Backend Architecture</h2>
 * <p>
 * Backends are responsible for:
 * </p>
 * <ul>
 *   <li><strong>Tensor Creation:</strong> Allocating native memory for tensors</li>
 *   <li><strong>Operation Execution:</strong> Performing arithmetic and activation functions</li>
 *   <li><strong>Memory Management:</strong> Managing device-specific memory pools</li>
 *   <li><strong>Device Coordination:</strong> Handling CPU/GPU data transfers</li>
 * </ul>
 * <p>
 * The runtime selects backends dynamically per-operation via {@link BackendRegistry},
 * enabling heterogeneous execution across different hardware accelerators.
 * </p>
 * <p>
 * <h2>Implementation Requirements</h2>
 * <p>
 * Backend implementations must:
 * </p>
 * <ul>
 *   <li>Be thread-safe for concurrent operation execution</li>
 *   <li>Handle their own native resource cleanup</li>
 *   <li>Support the standard tensor operations defined in this interface</li>
 *   <li>Register themselves with {@link BackendRegistry} at startup</li>
 * </ul>
 * <p>
 * <h2>Example Implementation</h2>
 * <pre>{@code
 * public class LibTorchBackend implements Backend {
 *     {@literal @Override}
 *     public BackendType type() {
 *         return BackendType.LIBTORCH;
 *     }
 *     
 *     {@literal @Override}
 *     public Tensor add(Tensor a, Tensor b, ExecutionContext ctx) {
 *         // Delegate to libtorch native library
 *         return LibTorchNative.add(a, b);
 *     }
 *     
 *     // ... other operations
 * }
 * }</pre>
 *
 * @see BackendRegistry
 * @see BackendType
 * @see Tensor
 * @since 1.0
 */
public interface Backend {

    /**
     * Returns the backend type identifier for this implementation.
     * <p>
     * The type is used by {@link BackendRegistry} to look up and route
     * operations to the correct backend implementation.
     * </p>
     *
     * @return the backend type identifier
     */
    BackendType type();

    // ── Arithmetic operations ─────────────────────────────────────────

    /**
     * Performs element-wise addition: {@code result = a + b}.
     * <p>
     * Both input tensors must have compatible shapes for broadcasting
     * and reside on devices supported by this backend.
     * </p>
     *
     * @param a the left operand
     * @param b the right operand
     * @param ctx the execution context for memory management
     * @return a new tensor containing the element-wise sum
     * @throws IllegalArgumentException if shapes are incompatible
     */
    Tensor add(Tensor a, Tensor b, ExecutionContext ctx);

    /**
     * Performs matrix multiplication: {@code result = a @ b}.
     * <p>
     * For 2D tensors, computes the standard matrix product.
     * For higher-dimensional tensors, performs batch matrix multiplication
     * over the leading dimensions.
     * </p>
     *
     * @param a the left operand
     * @param b the right operand
     * @param ctx the execution context for memory management
     * @return a new tensor containing the matrix product
     * @throws IllegalArgumentException if shapes are incompatible for matmul
     */
    Tensor matmul(Tensor a, Tensor b, ExecutionContext ctx);

    /**
     * Applies the ReLU activation function: {@code result = max(0, a)}.
     * <p>
     * Computes the Rectified Linear Unit element-wise, commonly used
     * in neural network inference.
     * </p>
     *
     * @param a the input tensor
     * @param ctx the execution context for memory management
     * @return a new tensor with ReLU applied
     */
    Tensor relu(Tensor a, ExecutionContext ctx);

    // ── Factory ───────────────────────────────────────────────────────

    /**
     * Creates a new tensor with the specified shape, dtype, and device.
     * <p>
     * The tensor's contents are uninitialized. Callers must explicitly
     * initialize the tensor before reading from it.
     * </p>
     * <p>
     * The backend allocates native memory appropriate for the target device
     * (e.g., GPU memory for CUDA tensors, system memory for CPU tensors).
     * </p>
     *
     * @param shape the tensor shape defining size along each dimension
     * @param dtype the data type for tensor elements
     * @param device the target compute device
     * @param ctx the execution context for memory management
     * @return a new tensor with uninitialized contents
     * @throws IllegalArgumentException if shape is invalid
     */
    Tensor createTensor(long[] shape, DType dtype, Device device, ExecutionContext ctx);

    // ── In-place operations (for memory planner) ──────────────────────

    /**
     * Performs element-wise addition into pre-allocated output memory.
     * <p>
     * This variant allows the memory planner to reuse existing allocations,
     * reducing memory pressure during graph execution. The {@code out} segment
     * must have sufficient capacity for the result.
     * </p>
     * <p>
     * Default implementation falls back to standard {@link #add(Tensor, Tensor, ExecutionContext)}
     * and may not provide in-place optimization.
     * </p>
     *
     * @param out pre-allocated output memory segment
     * @param a the left operand
     * @param b the right operand
     * @param ctx the execution context
     * @return a tensor wrapping the output memory
     */
    default Tensor addInto(java.lang.foreign.MemorySegment out, Tensor a, Tensor b, ExecutionContext ctx) {
        return add(a, b, ctx);
    }

    /**
     * Applies ReLU into pre-allocated output memory.
     * <p>
     * This variant allows the memory planner to reuse existing allocations.
     * The {@code out} segment must have sufficient capacity for the result.
     * </p>
     * <p>
     * Default implementation falls back to standard {@link #relu(Tensor, ExecutionContext)}.
     * </p>
     *
     * @param out pre-allocated output memory segment
     * @param a the input tensor
     * @param ctx the execution context
     * @return a tensor wrapping the output memory
     */
    default Tensor reluInto(java.lang.foreign.MemorySegment out, Tensor a, ExecutionContext ctx) {
        return relu(a, ctx);
    }

    /**
     * Performs matrix multiplication into pre-allocated output memory.
     * <p>
     * This variant allows the memory planner to reuse existing allocations.
     * The {@code out} segment must have sufficient capacity for the result.
     * </p>
     * <p>
     * Default implementation falls back to standard {@link #matmul(Tensor, Tensor, ExecutionContext)}.
     * </p>
     *
     * @param out pre-allocated output memory segment
     * @param a the left operand
     * @param b the right operand
     * @param ctx the execution context
     * @return a tensor wrapping the output memory
     */
    default Tensor matmulInto(java.lang.foreign.MemorySegment out, Tensor a, Tensor b, ExecutionContext ctx) {
        return matmul(a, b, ctx);
    }
}
