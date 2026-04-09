package tech.kayys.gollek.runtime.tensor;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Default concrete implementation of the {@link Tensor} interface for the Gollek inference runtime.
 * <p>
 * This class provides a complete tensor abstraction that wraps a {@link PooledTensorStorage} and
 * delegates computational operations to the appropriate {@link Backend} implementation (e.g., LibTorch,
 * GGML, ONNX Runtime, or LiteRT). The design emphasizes zero-copy view operations and efficient
 * memory management through reference-counted storage pooling.
 * <h2>Key Features</h2>
 * <ul>
 *   <li><strong>Zero-Copy Views:</strong> Reshape, slice, squeeze, and unsqueeze operations create
 *       new tensor views without copying underlying data, using offset and stride metadata.</li>
 *   <li><strong>Reference-Counted Memory:</strong> Multiple views can share the same storage with
 *       automatic memory reclamation when all references are closed.</li>
 *   <li><strong>Quantization Support:</strong> Native support for quantized dtypes (QINT8, QINT4)
 *       with per-tensor or per-group quantization parameters.</li>
 *   <li><strong>Backend Agnostic:</strong> Operations delegate to the appropriate backend based on
 *       the tensor's backend type, enabling heterogeneous execution.</li>
 * </ul>
 * <h2>Memory Layout</h2>
 * <p>
 * Tensors use a strided memory layout defined by:
 * </p>
 * <pre>
 *   element_index = offset + Σ(shape_index[i] × stride[i])
 * </pre>
 * <p>
 * For contiguous tensors in row-major order, the stride is computed as:
 * {@code stride[i] = stride[i+1] × shape[i+1]} with {@code stride[last] = 1}.
 * </p>
 * <h2>Thread Safety</h2>
 * <p>
 * This class is not thread-safe. Concurrent modifications to the same tensor instance
 * must be externally synchronized. However, multiple threads may safely read from
 * different tensor instances that share underlying storage.
 * </p>
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Create a contiguous tensor
 * PooledTensorStorage storage = ...;
 * Tensor tensor = new DefaultTensor(storage, new long[]{2, 3}, DType.FLOAT32, Device.CPU);
 *
 * // Create a view via reshape (zero-copy)
 * Tensor reshaped = tensor.reshape(6);
 *
 * // Slice along dimension 0
 * Tensor sliced = tensor.slice(0, 0, 1);
 *
 * // Operations delegate to backend
 * Tensor result = tensor.matmul(other, executionContext);
 *
 * // Release resources when done
 * tensor.close();
 * reshaped.close();  // Storage freed only when all views are closed
 * }</pre>
 *
 * @see Tensor
 * @see PooledTensorStorage
 * @see Backend
 * @since 1.0
 */
public class DefaultTensor implements Tensor {

    /**
     * Underlying pooled storage that manages the native memory segment.
     * Multiple tensors (views) may share the same storage instance.
     */
    private final PooledTensorStorage storage;

    /**
     * Shape of the tensor defining the size along each dimension.
     * For example, {@code [batch, sequence, hidden]} for a 3D tensor.
     */
    private final long[] shape;

    /**
     * Stride per dimension in elements (not bytes).
     * Defines the number of elements to skip to move to the next position
     * along each dimension. For contiguous row-major tensors:
     * {@code stride[i] = product(shape[i+1...end])}.
     */
    private final long[] stride;

    /**
     * Offset in elements from the base address of the storage.
     * Used for view tensors to point to a subset of the data without copying.
     * The actual byte offset is {@code offset × dtype.elementBytes()}.
     */
    private final long offset;

    /**
     * Data type of the tensor elements (e.g., FLOAT32, INT8, QINT4).
     * Determines memory layout and supported operations.
     */
    private final DType dtype;

    /**
     * Target compute device for this tensor (CPU, CUDA, Metal, etc.).
     * Operations may require tensors to be on the same device.
     */
    private final Device device;

    /**
     * Quantization parameters for quantized tensors (QINT8, QINT4).
     * Contains scale and zero_point values for dequantization.
     * Null for non-quantized dtypes.
     */
    private final QuantParams quantParams;

    /**
     * Lifecycle state flag. Once closed, the tensor cannot be used
     * and its storage reference count is decremented.
     */
    private boolean closed = false;

    /**
     * Creates a contiguous tensor with default row-major stride.
     * <p>
     * This is the primary constructor for creating new tensors with contiguous
     * memory layout. The stride is automatically computed to ensure row-major
     * ordering (last dimension has stride 1).
     *
     * @param storage underlying pooled storage managing native memory
     * @param shape tensor shape defining size along each dimension; must not be null
     * @param dtype data type of tensor elements; must not be null
     * @param device target compute device; must not be null
     * @throws IllegalArgumentException if shape contains negative dimensions
     * @throws NullPointerException if shape, dtype, or device is null
     */
    public DefaultTensor(
        PooledTensorStorage storage,
        long[] shape,
        DType dtype,
        Device device
    ) {
        this(storage, shape, computeContiguousStride(shape), 0, dtype, device, null);
    }

    /**
     * Creates a tensor with custom stride and offset for advanced memory layouts.
     * <p>
     * This constructor supports non-contiguous tensors and view creation where
     * the stride pattern differs from standard row-major ordering. Use this for
     * creating transposed views, sliced views, or other strided access patterns.
     *
     * @param storage underlying pooled storage managing native memory
     * @param shape tensor shape defining size along each dimension; must not be null
     * @param stride stride per dimension in elements; must match shape rank
     * @param offset offset in elements from storage base address; must be non-negative
     * @param dtype data type of tensor elements; must not be null
     * @param device target compute device; must not be null
     * @throws IllegalArgumentException if shape and stride ranks don't match,
     *         or if offset or any dimension is negative
     * @throws NullPointerException if shape, stride, dtype, or device is null
     */
    public DefaultTensor(
        PooledTensorStorage storage,
        long[] shape,
        long[] stride,
        long offset,
        DType dtype,
        Device device
    ) {
        this(storage, shape, stride, offset, dtype, device, null);
    }

    /**
     * Creates a tensor with custom stride, offset, and optional quantization parameters.
     * <p>
     * This is the most general constructor supporting all tensor configurations
     * including quantized formats. For quantized tensors (QINT8, QINT4), provide
     * appropriate quantization parameters for dequantization during computation.
     *
     * @param storage underlying pooled storage managing native memory
     * @param shape tensor shape defining size along each dimension; must not be null
     * @param stride stride per dimension in elements; must match shape rank
     * @param offset offset in elements from storage base address; must be non-negative
     * @param dtype data type of tensor elements; must not be null
     * @param device target compute device; must not be null
     * @param quantParams quantization parameters (scale, zero_point); may be null for non-quantized dtypes
     * @throws IllegalArgumentException if shape and stride ranks don't match,
     *         or if offset or any dimension is negative
     * @throws NullPointerException if shape, stride, dtype, or device is null
     */
    public DefaultTensor(
        PooledTensorStorage storage,
        long[] shape,
        long[] stride,
        long offset,
        DType dtype,
        Device device,
        QuantParams quantParams
    ) {
        validateTensor(shape, stride, offset, dtype);
        this.storage = storage;
        this.shape = shape.clone();
        this.stride = stride.clone();
        this.offset = offset;
        this.dtype = dtype;
        this.device = device;
        this.quantParams = quantParams;
    }

    /**
     * Creates a view tensor that shares storage with an existing tensor.
     * <p>
     * This private constructor is used internally to create zero-copy views
     * (reshape, slice, squeeze, etc.). When {@code isView} is true, the storage
     * reference count is incremented to ensure proper memory management.
     * <p>
     * <strong>Memory Management:</strong> Each view retains the storage, incrementing
     * its reference count. The underlying native memory is only released when all
     * views and the original tensor are closed.
     *
     * @param storage shared storage instance
     * @param shape view shape
     * @param stride view stride pattern
     * @param offset view offset from storage base
     * @param dtype data type (may differ from original for certain casts)
     * @param device target device
     * @param quantParams quantization parameters for the view
     * @param isView if true, increments storage reference count
     */
    private DefaultTensor(
        PooledTensorStorage storage,
        long[] shape,
        long[] stride,
        long offset,
        DType dtype,
        Device device,
        QuantParams quantParams,
        boolean isView
    ) {
        validateTensor(shape, stride, offset, dtype);
        this.storage = storage;
        this.shape = shape.clone();
        this.stride = stride.clone();
        this.offset = offset;
        this.dtype = dtype;
        this.device = device;
        this.quantParams = quantParams;
        if (isView) {
            storage.retain();
        }
    }

    // ── Validation ──────────────────────────────────────────────────────

    /**
     * Validates tensor construction parameters for correctness.
     * <p>
     * Performs the following checks:
     * <ul>
     *   <li>Null checks for shape, stride, and dtype</li>
     *   <li>Rank consistency: shape and stride must have same length</li>
     *   <li>Non-negative offset</li>
     *   <li>Non-negative shape dimensions</li>
     *   <li>Non-negative stride values</li>
     * </ul>
     *
     * @param shape tensor shape to validate
     * @param stride tensor stride to validate
     * @param offset tensor offset to validate
     * @param dtype tensor dtype to validate
     * @throws IllegalArgumentException if any validation fails
     * @throws NullPointerException if any required parameter is null
     */
    private void validateTensor(long[] shape, long[] stride, long offset, DType dtype) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        Objects.requireNonNull(stride, "Stride cannot be null");
        Objects.requireNonNull(dtype, "DType cannot be null");

        if (shape.length != stride.length) {
            throw new IllegalArgumentException(
                "Shape rank (" + shape.length + ") must match stride rank (" + stride.length + ")"
            );
        }

        if (offset < 0) {
            throw new IllegalArgumentException("Offset cannot be negative: " + offset);
        }

        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 0) {
                throw new IllegalArgumentException("Shape dimension " + i + " cannot be negative: " + shape[i]);
            }
        }

        for (int i = 0; i < stride.length; i++) {
            if (stride[i] < 0) {
                throw new IllegalArgumentException("Stride dimension " + i + " cannot be negative: " + stride[i]);
            }
        }
    }

    /**
     * Checks if the tensor has been closed and throws an exception if so.
     * <p>
     * This method should be called at the beginning of all operations that
     * access native memory to prevent use-after-free errors.
     *
     * @throws IllegalStateException if the tensor has been closed
     */
    private void checkClosed() {
        if (closed) {
            throw new IllegalStateException("Tensor has been closed");
        }
    }

    // ── Metadata ──────────────────────────────────────────────────────

    @Override
    public long[] shape() {
        checkClosed();
        return shape.clone();
    }

    @Override
    public DType dtype() {
        return dtype;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public BackendType backend() {
        return storage.backend();
    }

    @Override
    public long numel() {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        return n;
    }

    @Override
    public MemorySegment nativeHandle() {
        checkClosed();
        return storage.handle();
    }

    /**
     * Returns the storage offset in elements from the base address.
     * <p>
     * For contiguous tensors starting at the base address, this returns 0.
     * For view tensors (slices, reshapes), this returns the offset to the
     * first element accessible through this tensor view.
     * <p>
     * The byte offset can be computed as: {@code offset × dtype.elementBytes()}.
     *
     * @return offset in elements from storage base
     */
    public long offset() {
        return offset;
    }

    /**
     * Returns the stride per dimension in elements.
     * <p>
     * The stride array defines how many elements to skip in memory to move
     * to the next position along each dimension. For a contiguous row-major
     * tensor with shape {@code [d0, d1, d2]}, the stride is:
     * {@code [d1×d2, d2, 1]}.
     * <p>
     * Non-contiguous tensors (e.g., transposed views) may have different
     * stride patterns that don't follow this formula.
     *
     * @return stride array copy; modifications have no effect
     */
    public long[] stride() {
        return stride.clone();
    }

    @Override
    public boolean isQuantized() {
        return dtype.isQuantized();
    }

    @Override
    public QuantParams quantParams() {
        return quantParams;
    }

    // ── Operations ────────────────────────────────────────────────────

    /**
     * Retrieves the backend implementation for this tensor's backend type.
     * <p>
     * The backend is responsible for executing actual tensor computations.
     * This method looks up the appropriate backend from the registry based
     * on the tensor's backend type.
     *
     * @return backend implementation for executing operations
     * @see BackendRegistry
     * @see Backend
     */
    private Backend backendImpl() {
        return BackendRegistry.get(backend());
    }

    @Override
    public Tensor add(Tensor other, ExecutionContext ctx) {
        checkClosed();
        return backendImpl().add(this, other, ctx);
    }

    @Override
    public Tensor matmul(Tensor other, ExecutionContext ctx) {
        checkClosed();
        return backendImpl().matmul(this, other, ctx);
    }

    @Override
    public Tensor relu(ExecutionContext ctx) {
        checkClosed();
        return backendImpl().relu(this, ctx);
    }

    // ── View operations (zero-copy) ───────────────────────────────────

    /**
     * Creates a new view of this tensor with a different shape.
     * <p>
     * Reshape changes the tensor's shape without copying the underlying data.
     * The new shape must have the same total number of elements as the original.
     * The returned tensor is a view that shares the same storage, so modifications
     * to either tensor may affect the other.
     * <p>
     * <strong>Note:</strong> For non-contiguous tensors, this implementation
     * computes a contiguous stride for the new shape, which may not preserve
     * the original memory layout semantics. Advanced use cases may require
     * explicit transpose or permute operations before reshaping.
     *
     * @param newShape the desired new shape; must have same total elements
     * @return a new tensor view with the specified shape
     * @throws IllegalArgumentException if new shape has different element count,
     *         or if any dimension is negative
     * @throws IllegalStateException if this tensor has been closed
     * @see #squeeze()
     * @see #unsqueeze(long)
     */
    @Override
    public Tensor reshape(long... newShape) {
        checkClosed();
        Objects.requireNonNull(newShape, "New shape cannot be null");

        // Validate that the new shape has the same number of elements
        long newNumel = 1;
        for (long dim : newShape) {
            if (dim < 0) {
                throw new IllegalArgumentException("Invalid shape dimension: " + dim);
            }
            newNumel *= dim;
        }

        if (newNumel != numel()) {
            throw new IllegalArgumentException(
                "Cannot reshape tensor with " + numel() + " elements into shape " +
                Arrays.toString(newShape) + " with " + newNumel + " elements"
            );
        }

        // For contiguous tensors, compute new contiguous stride
        // For non-contiguous views, this is a simplification - may need refinement
        long[] newStride = computeContiguousStride(newShape);

        return new DefaultTensor(storage, newShape, newStride, offset, dtype, device, quantParams, true);
    }

    /**
     * Removes all dimensions of size 1 from the tensor shape.
     * <p>
     * Squeeze eliminates singleton dimensions to simplify tensor operations.
     * For example, a tensor with shape {@code [1, 3, 1, 5]} becomes {@code [3, 5]}.
     * <p>
     * If the tensor has no dimensions of size 1, this method returns {@code this}
     * without creating a new view.
     *
     * @return a view with all singleton dimensions removed, or this tensor if none exist
     * @throws IllegalStateException if this tensor has been closed
     * @see #unsqueeze(long)
     */
    @Override
    public Tensor squeeze() {
        checkClosed();
        List<Long> newShapeList = new ArrayList<>();
        for (long dim : shape) {
            if (dim != 1) {
                newShapeList.add(dim);
            }
        }

        long[] newShape = newShapeList.stream().mapToLong(Long::longValue).toArray();

        // If no dimensions were removed, return this tensor
        if (newShape.length == shape.length) {
            return this;
        }

        // Compute new strides by removing strides for dimensions of size 1
        List<Long> newStrideList = new ArrayList<>();
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != 1) {
                newStrideList.add(stride[i]);
            }
        }

        long[] newStride = newStrideList.stream().mapToLong(Long::longValue).toArray();

        return new DefaultTensor(storage, newShape, newStride, offset, dtype, device, quantParams, true);
    }

    /**
     * Adds a dimension of size 1 at the specified position.
     * <p>
     * Unsqueeze expands the tensor's dimensionality by inserting a singleton
     * dimension. This is useful for broadcasting operations or adding batch
     * dimensions. For example, unsqueezing a tensor with shape {@code [3, 5]}
     * at dimension 0 produces shape {@code [1, 3, 5]}.
     * <p>
     * Negative indices are supported: {@code -1} inserts at the end,
     * {@code -2} before the last dimension, etc.
     *
     * @param dim the position to insert the new dimension; may be negative
     * @return a new view with an added dimension of size 1
     * @throws IllegalArgumentException if dim is out of range {@code [-rank-1, rank]}
     * @throws IllegalStateException if this tensor has been closed
     * @see #squeeze()
     */
    @Override
    public Tensor unsqueeze(long dim) {
        checkClosed();
        int normalizedDim = normalizeDimension((int) dim, shape.length + 1);

        long[] newShape = new long[shape.length + 1];
        long[] newStride = new long[stride.length + 1];

        int srcIdx = 0;
        for (int i = 0; i < newShape.length; i++) {
            if (i == normalizedDim) {
                newShape[i] = 1;
                newStride[i] = 1;
            } else {
                newShape[i] = shape[srcIdx];
                newStride[i] = stride[srcIdx];
                srcIdx++;
            }
        }

        return new DefaultTensor(storage, newShape, newStride, offset, dtype, device, quantParams, true);
    }

    /**
     * Splits the tensor into multiple views along a specified dimension.
     * <p>
     * Split divides the tensor into chunks of size {@code splitSize} along
     * the given dimension. The last chunk may be smaller if the dimension
     * size is not evenly divisible by {@code splitSize}.
     * <p>
     * Each returned tensor is a zero-copy view sharing the same storage.
     * For example, splitting a tensor with shape {@code [4, 6]} by size 2
     * along dimension 1 produces 3 tensors each with shape {@code [4, 2]}.
     *
     * @param splitSize the size of each split chunk; must be positive
     * @param dim the dimension along which to split; may be negative
     * @return list of tensor views representing the splits
     * @throws IllegalArgumentException if splitSize is non-positive or if
     *         the dimension has size 0
     * @throws IllegalStateException if this tensor has been closed
     */
    @Override
    public List<Tensor> split(long splitSize, long dim) {
        checkClosed();
        int normalizedDim = normalizeDimension((int) dim, shape.length);

        if (splitSize <= 0) {
            throw new IllegalArgumentException("Split size must be positive: " + splitSize);
        }

        long dimSize = shape[normalizedDim];
        if (dimSize == 0) {
            throw new IllegalArgumentException("Cannot split tensor with dimension size 0");
        }

        List<Tensor> result = new ArrayList<>();
        long numSplits = (dimSize + splitSize - 1) / splitSize;

        for (long i = 0; i < numSplits; i++) {
            long start = i * splitSize;
            long end = Math.min(start + splitSize, dimSize);

            long[] newShape = shape.clone();
            newShape[normalizedDim] = end - start;

            // Compute new offset and stride for the slice
            long newOffset = offset + (start * stride[normalizedDim]);

            result.add(new DefaultTensor(storage, newShape, stride, newOffset, dtype, device, quantParams, true));
        }

        return result;
    }

    /**
     * Creates a view of a slice along a specific dimension.
     * <p>
     * Slice extracts a contiguous subset of elements from the tensor along
     * the given dimension. The slice is defined by {@code [start, end)}
     * (start inclusive, end exclusive).
     * <p>
     * For example, slicing a tensor with shape {@code [4, 6]} along
     * dimension 0 from 1 to 3 produces a view with shape {@code [2, 6]}.
     * <p>
     * The returned tensor is a zero-copy view that shares storage with
     * the original tensor.
     *
     * @param dim the dimension to slice along; may be negative
     * @param start the starting index (inclusive); must be in range {@code [0, size)}
     * @param end the ending index (exclusive); must be in range {@code (start, size]}
     * @return a new tensor view representing the sliced region
     * @throws IndexOutOfBoundsException if start or end are out of bounds
     * @throws IllegalStateException if this tensor has been closed
     */
    @Override
    public Tensor slice(int dim, long start, long end) {
        checkClosed();
        int normalizedDim = normalizeDimension(dim, shape.length);

        if (start < 0 || start >= shape[normalizedDim]) {
            throw new IndexOutOfBoundsException(
                "Start index " + start + " out of bounds for dimension " + normalizedDim +
                " with size " + shape[normalizedDim]
            );
        }

        if (end <= start || end > shape[normalizedDim]) {
            throw new IndexOutOfBoundsException(
                "End index " + end + " out of bounds for dimension " + normalizedDim +
                " with size " + shape[normalizedDim]
            );
        }

        long[] newShape = shape.clone();
        newShape[normalizedDim] = end - start;

        // Calculate new offset based on the slice position
        long newOffset = offset + (start * stride[normalizedDim]);

        return new DefaultTensor(storage, newShape, stride, newOffset, dtype, device, quantParams, true);
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    /**
     * Releases this tensor's reference to the underlying storage.
     * <p>
     * When the reference count reaches zero (i.e., all views sharing the
     * storage are also closed), the native memory is returned to the
     * {@link TensorPool} for reuse rather than being freed immediately.
     * <p>
     * This method is idempotent - calling it multiple times has no additional
     * effect. After closing, any attempt to access the tensor's data or
     * perform operations will throw {@link IllegalStateException}.
     * <p>
     * <strong>Important:</strong> All views created from this tensor must
     * also be closed independently. The underlying storage is only reclaimed
     * when the last reference is closed.
     */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            storage.release();
        }
    }

    /**
     * Checks whether this tensor has been closed.
     * <p>
     * This method is primarily useful for debugging and testing purposes.
     * Production code should rely on exception handling from {@link #checkClosed()}.
     *
     * @return true if the tensor has been closed, false otherwise
     */
    public boolean isClosed() {
        return closed;
    }

    // ── Utilities ─────────────────────────────────────────────────────

    /**
     * Computes row-major contiguous stride for a given shape.
     * <p>
     * In row-major (C-style) memory layout, the last dimension has stride 1,
     * and each preceding dimension's stride is the product of all succeeding
     * dimensions. For example, shape {@code [2, 3, 4]} produces stride
     * {@code [12, 4, 1]}.
     *
     * @param shape tensor shape
     * @return stride array in row-major order
     */
    private static long[] computeContiguousStride(long[] shape) {
        long[] stride = new long[shape.length];
        if (shape.length > 0) {
            stride[shape.length - 1] = 1;
            for (int i = shape.length - 2; i >= 0; i--) {
                stride[i] = stride[i + 1] * shape[i + 1];
            }
        }
        return stride;
    }

    /**
     * Normalizes a dimension index to handle negative indices (Python-style).
     * <p>
     * Negative indices count from the end: {@code -1} refers to the last
     * dimension, {@code -2} to the second-to-last, etc. This method converts
     * such indices to their positive equivalents.
     *
     * @param dim dimension index; may be negative
     * @param rank tensor rank (number of dimensions)
     * @return normalized dimension index in range {@code [0, rank)}
     * @throws IllegalArgumentException if dim is out of range {@code [-rank, rank)}
     */
    private static int normalizeDimension(int dim, int rank) {
        if (dim < 0) {
            dim = rank + dim;
        }
        if (dim < 0 || dim >= rank) {
            throw new IllegalArgumentException(
                "Dimension " + (dim - rank < 0 ? dim : dim) + " out of range for rank " + rank
            );
        }
        return dim;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof DefaultTensor)) return false;
        DefaultTensor other = (DefaultTensor) obj;
        return Arrays.equals(shape, other.shape)
            && Arrays.equals(stride, other.stride)
            && offset == other.offset
            && dtype == other.dtype
            && device == other.device
            && backend() == other.backend();
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(dtype, device, backend(), offset);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(stride);
        return result;
    }

    @Override
    public String toString() {
        return "Tensor[shape=" + Arrays.toString(shape)
            + ", dtype=" + dtype
            + ", device=" + device
            + ", backend=" + backend()
            + ", offset=" + offset
            + ", closed=" + closed + "]";
    }
}
