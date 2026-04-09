package tech.kayys.gollek.runtime.tensor;

import java.lang.foreign.MemorySegment;

/**
 * Universal tensor contract for the Gollek inference runtime — engine-agnostic.
 * <p>
 * This interface defines the core abstraction for multi-dimensional arrays used
 * throughout the inference pipeline. All inference engines (LibTorch, GGML, ONNX
 * Runtime, LiteRT) produce and consume tensors through this interface, with
 * operations delegating to the owning {@link Backend} transparently.
 * <p>
 * <h2>Key Characteristics</h2>
 * <ul>
 *   <li><strong>Backend Agnostic:</strong> Operations delegate to the appropriate
 *       backend (LibTorch, GGML, ONNX, etc.) based on the tensor's backend type.</li>
 *   <li><strong>Memory Management:</strong> Tensors are {@link AutoCloseable}.
 *       When closed, native memory is either freed or returned to a {@link TensorPool}
 *       for reuse, enabling zero-allocation execution during graph inference.</li>
 *   <li><strong>Zero-Copy Views:</strong> View operations (reshape, slice, squeeze)
 *       create new tensor instances without copying underlying data.</li>
 *   <li><strong>Quantization Support:</strong> Native support for quantized formats
 *       (QINT8, QINT4) with per-tensor or per-group quantization parameters.</li>
 * </ul>
 * <p>
 * <h2>Memory Layout</h2>
 * <p>
 * Tensors use a strided memory layout where each dimension has an associated stride
 * defining the number of elements to skip to advance to the next position. The actual
 * memory access is managed by the backend implementation.
 * </p>
 * <p>
 * <h2>Thread Safety</h2>
 * <p>
 * Tensor implementations are not required to be thread-safe. Concurrent access to
 * the same tensor instance must be externally synchronized. However, different tensor
 * instances may be accessed concurrently even if they share underlying storage.
 * </p>
 * <p>
 * <h2>Example Usage</h2>
 * <pre>{@code
 * try (ExecutionContext ctx = new ExecutionContext()) {
 *     // Create tensors through backend
 *     Tensor a = backend.createTensor(new long[]{2, 3}, DType.FLOAT32, Device.CPU, ctx);
 *     Tensor b = backend.createTensor(new long[]{3, 4}, DType.FLOAT32, Device.CPU, ctx);
 *     
 *     // Matrix multiplication
 *     Tensor result = a.matmul(b, ctx);
 *     
 *     // Apply activation
 *     Tensor activated = result.relu(ctx);
 *     
 *     // Create a view without copying
 *     Tensor sliced = activated.slice(0, 0, 1);
 *     
 *     // All tensors automatically cleaned up when context closes
 * }
 * }</pre>
 *
 * @see Backend
 * @see DType
 * @see Device
 * @see ExecutionContext
 * @see TensorPool
 * @since 1.0
 */
public interface Tensor extends AutoCloseable {

    // ── Metadata ──────────────────────────────────────────────────────

    /**
     * Returns the shape of this tensor defining its size along each dimension.
     * <p>
     * The shape array length represents the tensor's rank (number of dimensions).
     * Common shapes include:
     * </p>
     * <ul>
     *   <li>{@code [batch_size]} — 1D tensor (vector)</li>
     *   <li>{@code [batch_size, sequence_length]} — 2D tensor (matrix)</li>
     *   <li>{@code [batch_size, sequence_length, hidden_dim]} — 3D tensor</li>
     *   <li>{@code [batch_size, channels, height, width]} — 4D tensor (image)</li>
     * </ul>
     *
     * @return shape array copy; modifications have no effect
     */
    long[] shape();

    /**
     * Returns the data type of this tensor's elements.
     * <p>
     * The dtype determines memory layout, precision, and supported operations.
     * Common types include:
     * </p>
     * <ul>
     *   <li>{@link DType#FLOAT32} — Standard 32-bit floating point</li>
     *   <li>{@link DType#FLOAT16} — Half precision for memory efficiency</li>
     *   <li>{@link DType#BFLOAT16} — Brain floating point for ML workloads</li>
     *   <li>{@link DType#INT8} — 8-bit integer for quantized models</li>
     *   <li>{@link DType#QINT8} — Quantized 8-bit with scale/zero-point</li>
     * </ul>
     *
     * @return the data type
     */
    DType dtype();

    /**
     * Returns the compute device where this tensor's data resides.
     * <p>
     * Operations typically require tensors to be on the same device.
     * Cross-device operations may require explicit data transfer.
     * </p>
     *
     * @return the target device (CPU, CUDA, Metal, etc.)
     */
    Device device();

    /**
     * Returns the backend type that owns this tensor's memory.
     * <p>
     * The backend determines which native library executes operations
     * on this tensor. Common backends include LibTorch, GGML, and ONNX Runtime.
     * </p>
     *
     * @return the backend type
     * @see BackendType
     */
    BackendType backend();

    /**
     * Returns the total number of elements in this tensor.
     * <p>
     * This is the product of all shape dimensions. For example:
     * </p>
     * <ul>
     *   <li>Shape {@code [2, 3, 4]} → {@code 2 × 3 × 4 = 24} elements</li>
     *   <li>Shape {@code [5]} → {@code 5} elements</li>
     *   <li>Shape {@code []} (scalar) → {@code 1} element</li>
     * </ul>
     *
     * @return total element count
     */
    long numel();

    // ── Operations (delegate to backend) ──────────────────────────────

    /**
     * Performs element-wise addition: {@code result = this + other}.
     * <p>
     * Both tensors must have compatible shapes for broadcasting and be on
     * the same device. The result is a new tensor allocated by the backend.
     * </p>
     *
     * @param other the tensor to add; must have compatible shape
     * @param ctx the execution context for memory management
     * @return a new tensor containing the element-wise sum
     * @throws IllegalArgumentException if shapes are incompatible
     * @throws IllegalStateException if either tensor is closed
     */
    Tensor add(Tensor other, ExecutionContext ctx);

    /**
     * Performs matrix multiplication: {@code result = this @ other}.
     * <p>
     * For 2D tensors, this computes the standard matrix product where:
     * {@code result[i,j] = sum(this[i,k] * other[k,j])}.
     * </p>
     * <p>
     * For higher-dimensional tensors, batch matrix multiplication is performed
     * over the leading dimensions. The last two dimensions are treated as
     * the matrix dimensions.
     * </p>
     *
     * @param other the right-hand operand; must have compatible shape
     * @param ctx the execution context for memory management
     * @return a new tensor containing the matrix product
     * @throws IllegalArgumentException if shapes are incompatible for matmul
     * @throws IllegalStateException if either tensor is closed
     */
    Tensor matmul(Tensor other, ExecutionContext ctx);

    /**
     * Applies the ReLU (Rectified Linear Unit) activation function.
     * <p>
     * Computes {@code result = max(0, this)} element-wise. ReLU is commonly
     * used in neural networks to introduce non-linearity.
     * </p>
     * <p>
     * Implementations may perform the operation in-place if possible to
     * conserve memory.
     * </p>
     *
     * @param ctx the execution context for memory management
     * @return a new tensor with ReLU applied (or this tensor if in-place)
     * @throws IllegalStateException if this tensor is closed
     */
    Tensor relu(ExecutionContext ctx);

    // ── View operations (zero-copy) ───────────────────────────────────

    /**
     * Creates a new view of this tensor with a different shape.
     * <p>
     * Reshape changes the tensor's shape without copying the underlying data.
     * The new shape must have the same total number of elements as the original.
     * </p>
     *
     * @param newShape the desired new shape; must have same total elements
     * @return a new tensor view with the specified shape
     * @throws IllegalArgumentException if new shape has different element count
     * @throws IllegalStateException if this tensor is closed
     */
    Tensor reshape(long... newShape);

    /**
     * Removes all dimensions of size 1 from the tensor shape.
     * <p>
     * For example, a tensor with shape {@code [1, 3, 1, 5]} becomes {@code [3, 5]}.
     * </p>
     *
     * @return a view with all singleton dimensions removed
     * @throws IllegalStateException if this tensor is closed
     */
    Tensor squeeze();

    /**
     * Adds a dimension of size 1 at the specified position.
     * <p>
     * For example, unsqueezing a tensor with shape {@code [3, 5]} at
     * dimension 0 produces shape {@code [1, 3, 5]}.
     * </p>
     * <p>
     * Negative indices are supported: {@code -1} inserts at the end.
     * </p>
     *
     * @param dim the position to insert the new dimension; may be negative
     * @return a new view with an added dimension of size 1
     * @throws IllegalArgumentException if dim is out of range
     * @throws IllegalStateException if this tensor is closed
     */
    Tensor unsqueeze(long dim);

    /**
     * Splits the tensor into multiple views along a specified dimension.
     * <p>
     * Divides the tensor into chunks of size {@code splitSize} along the
     * given dimension. The last chunk may be smaller if the dimension size
     * is not evenly divisible.
     * </p>
     *
     * @param splitSize the size of each split chunk; must be positive
     * @param dim the dimension along which to split; may be negative
     * @return list of tensor views representing the splits
     * @throws IllegalArgumentException if splitSize is non-positive
     * @throws IllegalStateException if this tensor is closed
     */
    java.util.List<Tensor> split(long splitSize, long dim);

    /**
     * Creates a view of a slice along a specific dimension.
     * <p>
     * Extracts a contiguous subset of elements from the tensor along the
     * given dimension. The slice is defined by {@code [start, end)}
     * (start inclusive, end exclusive).
     * </p>
     *
     * @param dim the dimension to slice along; may be negative
     * @param start the starting index (inclusive)
     * @param end the ending index (exclusive)
     * @return a new tensor view representing the sliced region
     * @throws IndexOutOfBoundsException if start or end are out of bounds
     * @throws IllegalStateException if this tensor is closed
     */
    Tensor slice(int dim, long start, long end);

    // ── Native access ─────────────────────────────────────────────────

    /**
     * Returns the raw native memory handle for this tensor.
     * <p>
     * <strong>Warning:</strong> This method is for internal use by backends only.
     * Callers must NOT free this handle directly. Memory management is handled
     * automatically through the tensor's {@link #close()} method and
     * {@link TensorPool}.
     * </p>
     *
     * @return the native memory segment
     * @throws IllegalStateException if this tensor is closed
     */
    MemorySegment nativeHandle();

    // ── Quantization support ──────────────────────────────────────────

    /**
     * Returns whether this tensor uses a quantized data type.
     * <p>
     * Quantized tensors (QINT8, QINT4) store values as integers with
     * associated scale and zero-point parameters for dequantization.
     * </p>
     *
     * @return true if this tensor is quantized
     * @see DType#isQuantized()
     */
    default boolean isQuantized() {
        return dtype().isQuantized();
    }

    /**
     * Returns the quantization parameters for this tensor.
     * <p>
     * Quantization parameters define the mapping from quantized integer
     * values to real floating-point values:
     * </p>
     * <pre>
     *   real_value = (int_value - zeroPoint) × scale
     * </pre>
     * <p>
     * Returns {@code null} for non-quantized tensors.
     * </p>
     *
     * @return quantization parameters, or null if not quantized
     * @see QuantParams
     */
    default QuantParams quantParams() {
        return null;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    /**
     * Releases this tensor's resources.
     * <p>
     * When closed, the tensor's native memory is either freed or returned
     * to the {@link TensorPool} for reuse, depending on the memory management
     * strategy in use.
     * </p>
     * <p>
     * After closing, any attempt to access the tensor's data or perform
     * operations will throw {@link IllegalStateException}.
     * </p>
     */
    @Override
    void close();
}
