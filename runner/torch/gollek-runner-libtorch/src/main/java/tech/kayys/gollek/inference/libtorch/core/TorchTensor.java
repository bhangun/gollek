package tech.kayys.gollek.inference.libtorch.core;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.runtime.tensor.BackendType;
import tech.kayys.gollek.runtime.tensor.DType;
import tech.kayys.gollek.runtime.tensor.ExecutionContext;
import tech.kayys.gollek.runtime.tensor.Tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

/**
 * LibTorch-specific tensor implementation wrapping native at::Tensor via FFM.
 * <p>
 * This class implements the universal
 * {@link Tensor}
 * SPI interface, delegating operations to LibTorch native bindings.
 * <p>
 * Each tensor owns a confined {@link Arena} for its native memory. When closed,
 * the arena is released and the underlying native tensor is freed.
 * <p>
 * A {@link Cleaner}-based safety net ensures native memory is freed even if
 * the caller forgets to call {@link #close()}. A warning is logged when this
 * happens to help track resource leaks.
 * <p>
 * <h2>Thread Safety</h2>
 * <p>
 * This class is not thread-safe. Each tensor should be used from a single
 * thread.
 * For concurrent access, create separate tensor instances or synchronize
 * externally.
 * </p>
 * <p>
 * <h2>Memory Management</h2>
 * <p>
 * Tensors must be explicitly closed to release native memory:
 * </p>
 * 
 * <pre>{@code
 * try (TorchTensor tensor = TorchTensor.randn(2, 3)) {
 *     // Use tensor...
 *     TorchTensor result = tensor.matmul(other);
 * } // Native memory freed here
 * }</pre>
 * <p>
 * <h2>Leak Detection</h2>
 * <p>
 * Enable detailed leak tracking via system property:
 * {@code -Dgollek.tensor.leak-detection=true}
 * </p>
 *
 * @see Tensor
 * @see ScalarType
 * @see Device
 * @since 1.0
 */
public class TorchTensor implements Tensor {

    private static final System.Logger LEAK_LOGGER = System.getLogger("tech.kayys.gollek.tensor.leak");
    private static final AtomicLong LIVE_COUNT = new AtomicLong(0);

    /**
     * Enable detailed leak tracking with allocation stack traces.
     * Controlled via system property {@code gollek.tensor.leak-detection=true}.
     */
    private static final boolean LEAK_DETECTION = Boolean.getBoolean("gollek.tensor.leak-detection");

    private final MemorySegment nativeHandle;
    private final Arena arena;
    private boolean closed = false;
    private final Cleaner.Cleanable cleanable;
    private final Exception allocationSite;

    /**
     * Creates a tensor wrapper around an existing native handle.
     *
     * @param nativeHandle pointer to the native at::Tensor
     * @param arena        arena managing the native memory
     */
    public TorchTensor(MemorySegment nativeHandle, Arena arena) {
        this.nativeHandle = Objects.requireNonNull(nativeHandle, "nativeHandle must not be null");
        this.arena = Objects.requireNonNull(arena, "arena must not be null");
        this.allocationSite = LEAK_DETECTION ? new Exception("Tensor allocated here") : null;
        this.cleanable = CleanerHolder.CLEANER.register(this, new TensorCleaner(nativeHandle, arena, allocationSite));
        LIVE_COUNT.incrementAndGet();
    }

    /**
     * Returns the number of currently live (unclosed) tensors.
     * <p>
     * Useful for diagnostics and leak tracking.
     * </p>
     *
     * @return count of live tensors
     */
    public static long liveCount() {
        return LIVE_COUNT.get();
    }

    // ── Factory methods ───────────────────────────────────────────────

    /**
     * Creates an uninitialized tensor with the given shape and type.
     *
     * @param shape  the tensor shape
     * @param dtype  the scalar type
     * @param device the target device
     * @return a new uninitialized tensor
     */
    public static TorchTensor empty(long[] shape, ScalarType dtype, Device device) {
        return createTensor(LibTorchBinding.TENSOR_EMPTY, LibTorchBinding.TENSOR_EMPTY_DESC, shape, dtype, device);
    }

    /**
     * Creates a tensor filled with zeros.
     *
     * @param shape  the tensor shape
     * @param dtype  the scalar type
     * @param device the target device
     * @return a new tensor filled with zeros
     */
    public static TorchTensor zeros(long[] shape, ScalarType dtype, Device device) {
        return createTensor(LibTorchBinding.TENSOR_ZEROS, LibTorchBinding.TENSOR_ZEROS_DESC, shape, dtype, device);
    }

    /**
     * Creates a tensor filled with ones.
     *
     * @param shape  the tensor shape
     * @param dtype  the scalar type
     * @param device the target device
     * @return a new tensor filled with ones
     */
    public static TorchTensor ones(long[] shape, ScalarType dtype, Device device) {
        return createTensor(LibTorchBinding.TENSOR_ONES, LibTorchBinding.TENSOR_ONES_DESC, shape, dtype, device);
    }

    /**
     * Creates a tensor with random values from a normal distribution.
     *
     * @param shape  the tensor shape
     * @param dtype  the scalar type
     * @param device the target device
     * @return a new tensor with normal random values
     */
    public static TorchTensor randn(long[] shape, ScalarType dtype, Device device) {
        return createTensor(LibTorchBinding.TENSOR_RANDN, LibTorchBinding.TENSOR_RANDN_DESC, shape, dtype, device);
    }

    /**
     * Creates a tensor with random values from a uniform distribution [0, 1).
     *
     * @param shape  the tensor shape
     * @param dtype  the scalar type
     * @param device the target device
     * @return a new tensor with uniform random values
     */
    public static TorchTensor rand(long[] shape, ScalarType dtype, Device device) {
        return createTensor(LibTorchBinding.TENSOR_RAND, LibTorchBinding.TENSOR_RAND_DESC, shape, dtype, device);
    }

    /**
     * Convenience method to create a float tensor filled with zeros on CPU.
     *
     * @param shape the tensor shape
     * @return a new float tensor filled with zeros
     */
    public static TorchTensor zeros(long... shape) {
        return zeros(shape, ScalarType.FLOAT, Device.CPU);
    }

    /**
     * Convenience method to create a float tensor filled with ones on CPU.
     *
     * @param shape the tensor shape
     * @return a new float tensor filled with ones
     */
    public static TorchTensor ones(long... shape) {
        return ones(shape, ScalarType.FLOAT, Device.CPU);
    }

    /**
     * Convenience method to create a float tensor with normal random values on CPU.
     *
     * @param shape the tensor shape
     * @return a new float tensor with normal random values
     */
    public static TorchTensor randn(long... shape) {
        return randn(shape, ScalarType.FLOAT, Device.CPU);
    }

    /**
     * Creates a tensor from a float array.
     *
     * @param data  the source float array
     * @param shape the target tensor shape
     * @return a new tensor containing the data
     */
    public static TorchTensor fromFloatArray(float[] data, long[] shape) {
        return fromFloatArray(data, shape, null);
    }

    public static TorchTensor fromFloatArray(float[] data, long[] shape, Device device) {
        Arena arena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(LibTorchBinding.TENSOR_FROM_BLOB, LibTorchBinding.TENSOR_FROM_BLOB_DESC);

            MemorySegment dataSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, data);
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment result = (MemorySegment) fn.invoke(dataSegment, shapeSegment, (long) shape.length,
                    ScalarType.FLOAT.code());
            TorchTensor tensor = new TorchTensor(result, arena);

            if (device != null && device != Device.CPU) {
                tensor = tensor.to(device);
            }
            return tensor;
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to create tensor from float array", t);
        }
    }

    /**
     * Creates a tensor from a double array.
     *
     * @param data  the source double array
     * @param shape the target tensor shape
     * @return a new tensor containing the data
     */
    public static TorchTensor fromDoubleArray(double[] data, long[] shape) {
        Arena arena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(LibTorchBinding.TENSOR_FROM_BLOB, LibTorchBinding.TENSOR_FROM_BLOB_DESC);

            MemorySegment dataSegment = arena.allocateFrom(ValueLayout.JAVA_DOUBLE, data);
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment result = (MemorySegment) fn.invoke(dataSegment, shapeSegment, (long) shape.length,
                    ScalarType.DOUBLE.code());
            return new TorchTensor(result, arena);
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to create tensor from double array", t);
        }
    }

    /**
     * Creates a tensor from a long array (e.g., for index tensors).
     *
     * @param data  the source long array
     * @param shape the target tensor shape
     * @return a new tensor containing the data
     */
    public static TorchTensor fromLongArray(long[] data, long[] shape) {
        return fromLongArray(data, shape, null); // Default to current device
    }

    /**
     * Create a long tensor with optional device placement.
     * 
     * @param data   the data array
     * @param shape  the tensor shape
     * @param device the device to place tensor on (null = default)
     */
    public static TorchTensor fromLongArray(long[] data, long[] shape, Device device) {
        Arena arena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(LibTorchBinding.TENSOR_FROM_BLOB, LibTorchBinding.TENSOR_FROM_BLOB_DESC);

            MemorySegment dataSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, data);
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment result = (MemorySegment) fn.invoke(dataSegment, shapeSegment, (long) shape.length,
                    ScalarType.LONG.code());
            TorchTensor tensor = new TorchTensor(result, arena);

            // Move to specified device if not null
            if (device != null && device != Device.CPU) {
                tensor = tensor.to(device);
            }
            return tensor;
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to create tensor from long array", t);
        }
    }

    /**
     * Concatenates a list of tensors along the given dimension.
     * <p>
     * Used by the continuous batching manager to merge individual inputs into a
     * batch.
     * </p>
     *
     * @param tensors list of tensors to concatenate
     * @param dim     dimension along which to concatenate (typically 0 for
     *                batching)
     * @return a new tensor representing the concatenated result
     */
    public static TorchTensor cat(java.util.List<TorchTensor> tensors, long dim) {
        if (tensors == null || tensors.isEmpty()) {
            throw new IllegalArgumentException("Cannot concatenate empty tensor list");
        }
        Arena arena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(LibTorchBinding.TENSOR_CAT, LibTorchBinding.TENSOR_CAT_DESC);

            // Allocate an array of native pointers for the input tensors
            MemorySegment pointers = arena.allocate(ValueLayout.ADDRESS, tensors.size());
            for (int i = 0; i < tensors.size(); i++) {
                pointers.setAtIndex(ValueLayout.ADDRESS, i, tensors.get(i).nativeHandle());
            }

            MemorySegment result = (MemorySegment) fn.invoke(pointers, (long) tensors.size(), dim);
            return new TorchTensor(result, arena);
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to concatenate tensors", t);
        }
    }

    /**
     * Stacks a list of tensors along the given dimension.
     *
     * @param tensors list of tensors to stack
     * @param dim     the dimension along which to stack
     * @return a new tensor representing the stacked result
     */
    public static TorchTensor stack(java.util.List<TorchTensor> tensors, int dim) {
        if (tensors == null || tensors.isEmpty()) {
            throw new IllegalArgumentException("Cannot stack empty tensor list");
        }
        Arena arena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            java.lang.invoke.MethodHandle fn = binding.bind(LibTorchBinding.TENSOR_STACK,
                    LibTorchBinding.TENSOR_STACK_DESC);

            // Allocate an array of native pointers for the input tensors
            MemorySegment pointers = arena.allocate(ValueLayout.ADDRESS, tensors.size());
            for (int i = 0; i < tensors.size(); i++) {
                pointers.setAtIndex(ValueLayout.ADDRESS, i, tensors.get(i).nativeHandle());
            }

            MemorySegment result = (MemorySegment) fn.invoke(pointers, (long) tensors.size(), (long) dim);
            return new TorchTensor(result, arena);
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to stack tensors", t);
        }
    }

    /**
     * Selects elements along a dimension using an index tensor.
     * <p>
     * Used by the continuous batching manager to slice batched output back to
     * individual results.
     * </p>
     *
     * @param dim   dimension along which to select
     * @param index 1-D index tensor
     * @return a new tensor with the selected elements
     */
    public TorchTensor indexSelect(long dim, TorchTensor index) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(LibTorchBinding.TENSOR_INDEX_SELECT,
                    LibTorchBinding.TENSOR_INDEX_SELECT_DESC);

            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, dim, index.nativeHandle());
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("index_select failed", t);
        }
    }

    /**
     * Returns the value of this tensor as a standard Java long.
     * <p>
     * This only works for tensors with one element.
     * </p>
     *
     * @return the long value
     * @throws IllegalStateException if tensor has more than one element
     */
    public long itemLong() {
        checkClosed();
        if (numel() != 1) {
            throw new IllegalStateException("Tensor must have exactly one element to call itemLong()");
        }

        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle dataPtrFn = binding.bind(LibTorchBinding.TENSOR_DATA_PTR,
                    LibTorchBinding.TENSOR_DATA_PTR_DESC);

            MemorySegment ptr = (MemorySegment) dataPtrFn.invoke(nativeHandle);
            // Reinterpret as long buffer of size 8
            MemorySegment data = ptr.reinterpret(8);

            // Check dtype
            ScalarType currentDtype = scalarType();
            if (currentDtype == ScalarType.LONG) {
                return data.get(ValueLayout.JAVA_LONG, 0);
            } else if (currentDtype == ScalarType.INT) {
                return data.get(ValueLayout.JAVA_INT, 0);
            } else {
                throw new IllegalStateException("itemLong() only supports LONG or INT tensors, got: " + currentDtype);
            }
        } catch (Throwable t) {
            throw new RuntimeException("itemLong failed", t);
        }
    }

    private static TorchTensor createTensor(String fnName, java.lang.foreign.FunctionDescriptor desc,
            long[] shape, ScalarType dtype, Device device) {
        Arena arena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(fnName, desc);

            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment result = (MemorySegment) fn.invoke(shapeSegment, (long) shape.length, dtype.code(),
                    device.type().code());
            return new TorchTensor(result, arena);
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to create tensor via " + fnName, t);
        }
    }

    // ── Properties ────────────────────────────────────────────────────

    /**
     * Returns the number of dimensions.
     *
     * @return the tensor rank
     */
    public long dim() {
        checkClosed();
        return invokeUnaryLong(LibTorchBinding.TENSOR_DIM, LibTorchBinding.TENSOR_DIM_DESC);
    }

    /**
     * Returns the total number of elements.
     *
     * @return the element count
     */
    public long numel() {
        checkClosed();
        return invokeUnaryLong(LibTorchBinding.TENSOR_NUMEL, LibTorchBinding.TENSOR_NUMEL_DESC);
    }

    /**
     * Returns the size along a specific dimension.
     *
     * @param dim the dimension index
     * @return the size of that dimension
     */
    public long size(long dim) {
        checkClosed();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_SIZE,
                    LibTorchBinding.TENSOR_SIZE_DESC);
            return (long) fn.invoke(nativeHandle, dim);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get tensor size", t);
        }
    }

    /**
     * Returns the shape as an array.
     *
     * @return the tensor shape
     */
    public long[] shape() {
        checkClosed();
        int ndim = (int) dim();
        long[] result = new long[ndim];
        for (int i = 0; i < ndim; i++) {
            result[i] = size(i);
        }
        return result;
    }

    /**
     * Returns the scalar type (LibTorch specific).
     *
     * @return the scalar type
     */
    public ScalarType scalarType() {
        checkClosed();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_DTYPE,
                    LibTorchBinding.TENSOR_DTYPE_DESC);
            int code = (int) fn.invoke(nativeHandle);
            return ScalarType.fromCode(code);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get tensor dtype", t);
        }
    }

    // ── SPI Implementations ───────────────────────────────────────────

    @Override
    public DType dtype() {
        return switch (scalarType()) {
            case FLOAT -> DType.FLOAT32;
            case HALF -> DType.FLOAT16;
            case BFLOAT16 -> DType.BFLOAT16;
            case CHAR -> DType.INT8;
            case QUINT8, QINT8 -> DType.QINT8;
            default -> DType.FLOAT32;
        };
    }

    @Override
    public tech.kayys.gollek.runtime.tensor.Device device() {
        return switch (deviceType()) {
            case CPU -> tech.kayys.gollek.runtime.tensor.Device.CPU;
            case CUDA -> tech.kayys.gollek.runtime.tensor.Device.CUDA;
            case MPS -> tech.kayys.gollek.runtime.tensor.Device.METAL;
            case HIP -> tech.kayys.gollek.runtime.tensor.Device.ROCM;
            case XLA -> tech.kayys.gollek.runtime.tensor.Device.TPU;
            default -> tech.kayys.gollek.runtime.tensor.Device.CPU;
        };
    }

    @Override
    public BackendType backend() {
        return BackendType.LIBTORCH;
    }

    @Override
    public Tensor add(
            Tensor other,
            ExecutionContext ctx) {
        return this.add((TorchTensor) other);
    }

    @Override
    public Tensor matmul(
            Tensor other,
            ExecutionContext ctx) {
        return this.matmul((TorchTensor) other);
    }

    @Override
    public Tensor relu(
            ExecutionContext ctx) {
        return this.relu();
    }

    /**
     * Applies the ReLU activation function element-wise.
     *
     * @return a new tensor with ReLU applied
     */
    public TorchTensor relu() {
        return invokeUnaryOp(LibTorchBinding.NN_RELU);
    }

    @Override
    public tech.kayys.gollek.runtime.tensor.Tensor slice(int dim, long start, long end) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            Optional<MethodHandle> sliceFn = binding.bindOptional(
                    LibTorchBinding.TENSOR_SLICE, LibTorchBinding.TENSOR_SLICE_DESC);

            if (sliceFn.isPresent()) {
                MemorySegment result = (MemorySegment) sliceFn.get().invoke(
                        nativeHandle, (long) dim, start, end, 1L);
                return new TorchTensor(result, opArena);
            }

            throw new UnsupportedOperationException("at_slice native function not found in LibTorchBinding");
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("slice failed", t);
        }
    }

    /**
     * Returns the device type.
     *
     * @return the device type
     */
    public Device.Type deviceType() {
        checkClosed();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_DEVICE_TYPE,
                    LibTorchBinding.TENSOR_DEVICE_TYPE_DESC);
            int code = (int) fn.invoke(nativeHandle);
            return Device.Type.fromCode(code);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get tensor device type", t);
        }
    }

    // ── Data access ───────────────────────────────────────────────────

    /**
     * Returns the raw data pointer.
     *
     * @return the native data pointer
     */
    public MemorySegment dataPtr() {
        checkClosed();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_DATA_PTR,
                    LibTorchBinding.TENSOR_DATA_PTR_DESC);
            return (MemorySegment) fn.invoke(nativeHandle);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to get data pointer", t);
        }
    }

    /**
     * Copies tensor data to a float array.
     *
     * @return a new float array containing the tensor data
     */
    public float[] toFloatArray() {
        checkClosed();
        long n = numel();
        MemorySegment data = dataPtr().reinterpret(n * ValueLayout.JAVA_FLOAT.byteSize());
        return data.toArray(ValueLayout.JAVA_FLOAT);
    }

    /**
     * Copies tensor data to a double array.
     *
     * @return a new double array containing the tensor data
     */
    public double[] toDoubleArray() {
        checkClosed();
        long n = numel();
        MemorySegment data = dataPtr().reinterpret(n * ValueLayout.JAVA_DOUBLE.byteSize());
        return data.toArray(ValueLayout.JAVA_DOUBLE);
    }

    /**
     * Copies tensor data to a long array.
     *
     * @return a new long array containing the tensor data
     */
    public long[] toLongArray() {
        checkClosed();
        long n = numel();
        MemorySegment data = dataPtr().reinterpret(n * ValueLayout.JAVA_LONG.byteSize());
        return data.toArray(ValueLayout.JAVA_LONG);
    }

    // ── Binary operations ─────────────────────────────────────────────

    /**
     * Performs element-wise addition.
     *
     * @param other the tensor to add
     * @return a new tensor containing the sum
     */
    public TorchTensor add(TorchTensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_ADD, other);
    }

    /**
     * Performs element-wise subtraction.
     *
     * @param other the tensor to subtract
     * @return a new tensor containing the difference
     */
    public TorchTensor sub(TorchTensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_SUB, other);
    }

    /**
     * Performs element-wise multiplication.
     *
     * @param other the tensor to multiply with
     * @return a new tensor containing the product
     */
    public TorchTensor mul(TorchTensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_MUL, other);
    }

    /**
     * Performs element-wise division.
     *
     * @param other the tensor to divide by
     * @return a new tensor containing the quotient
     */
    public TorchTensor div(TorchTensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_DIV, other);
    }

    /**
     * Performs matrix multiplication.
     *
     * @param other the right-hand operand
     * @return a new tensor containing the matrix product
     */
    public TorchTensor matmul(TorchTensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_MATMUL, other);
    }

    // ── Comparison operations ─────────────────────────────────────────

    /**
     * Performs element-wise equality comparison.
     *
     * @param other the tensor to compare with
     * @return a boolean tensor with comparison results
     */
    public TorchTensor eq(TorchTensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_EQ, other);
    }

    /**
     * Performs element-wise greater-than comparison.
     *
     * @param other the tensor to compare with
     * @return a boolean tensor with comparison results
     */
    public TorchTensor gt(TorchTensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_GT, other);
    }

    /**
     * Performs element-wise less-than comparison.
     *
     * @param other the tensor to compare with
     * @return a boolean tensor with comparison results
     */
    public TorchTensor lt(TorchTensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_LT, other);
    }

    // ── Unary operations ──────────────────────────────────────────────

    /**
     * Negates all elements.
     *
     * @return a new tensor with negated values
     */
    public TorchTensor neg() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_NEG);
    }

    /**
     * Computes absolute value element-wise.
     *
     * @return a new tensor with absolute values
     */
    public TorchTensor abs() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_ABS);
    }

    /**
     * Computes square root element-wise.
     *
     * @return a new tensor with square roots
     */
    public TorchTensor sqrt() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_SQRT);
    }

    /**
     * Computes natural logarithm element-wise.
     *
     * @return a new tensor with logarithm values
     */
    public TorchTensor log() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_LOG);
    }

    /**
     * Computes exponential element-wise.
     *
     * @return a new tensor with exponential values
     */
    public TorchTensor exp() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_EXP);
    }

    /**
     * Applies softmax along a dimension.
     *
     * @param input the input tensor
     * @param dim   the dimension along which to apply softmax
     * @return a new tensor with softmax applied
     */
    public static TorchTensor softmax(TorchTensor input, long dim) {
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(LibTorchBinding.NN_SOFTMAX, LibTorchBinding.NN_SOFTMAX_DESC);
            Arena opArena = Arena.ofConfined();
            MemorySegment result = (MemorySegment) fn.invoke(input.nativeHandle(), dim);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to execute softmax", t);
        }
    }

    // ── Reduction operations ──────────────────────────────────────────

    /**
     * Computes the sum of all elements.
     *
     * @return a scalar tensor with the sum
     */
    public TorchTensor sum() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_SUM);
    }

    /**
     * Computes the mean of all elements.
     *
     * @return a scalar tensor with the mean
     */
    public TorchTensor mean() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_MEAN);
    }

    /**
     * Computes the maximum element.
     *
     * @return a scalar tensor with the maximum value
     */
    public TorchTensor max() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_MAX);
    }

    /**
     * Computes the minimum element.
     *
     * @return a scalar tensor with the minimum value
     */
    public TorchTensor min() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_MIN);
    }

    /**
     * Returns the index of the maximum element along a dimension.
     *
     * @param dim the dimension to reduce
     * @return a tensor with indices of maximum values
     */
    public TorchTensor argmax(long dim) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_ARGMAX,
                    LibTorchBinding.TENSOR_ARGMAX_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, dim);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to compute argmax", t);
        }
    }

    // ── Shape operations ──────────────────────────────────────────────

    /**
     * Reshapes the tensor to a new shape without changing data.
     *
     * @param newShape the desired new shape
     * @return a new tensor with the reshaped view
     */
    public TorchTensor reshape(long... newShape) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(LibTorchBinding.TENSOR_RESHAPE, LibTorchBinding.TENSOR_RESHAPE_DESC);
            MemorySegment shapeSegment = opArena.allocateFrom(ValueLayout.JAVA_LONG, newShape);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, shapeSegment, (long) newShape.length);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to reshape tensor", t);
        }
    }

    /**
     * Transposes two dimensions.
     *
     * @param dim0 the first dimension
     * @param dim1 the second dimension
     * @return a new tensor with transposed dimensions
     */
    public TorchTensor transpose(long dim0, long dim1) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_TRANSPOSE,
                    LibTorchBinding.TENSOR_TRANSPOSE_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, dim0, dim1);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to transpose tensor", t);
        }
    }

    /**
     * Squeezes all dimensions of size 1.
     *
     * @return a new tensor with singleton dimensions removed
     */
    public TorchTensor squeeze() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_SQUEEZE);
    }

    /**
     * Unsqueezes the tensor by adding a dimension of size 1 at the specified
     * position.
     *
     * @param dim the position to insert the new dimension
     * @return a new tensor with an added dimension
     */
    public TorchTensor unsqueeze(long dim) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_UNSQUEEZE,
                    LibTorchBinding.TENSOR_UNSQUEEZE_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, dim);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to unsqueeze tensor", t);
        }
    }

    /**
     * Splits the tensor into chunks of `splitSize` along `dim`.
     *
     * @param splitSize the size of each chunk
     * @param dim       the dimension along which to split
     * @return a list of tensor chunks
     */
    @Override
    public java.util.List<Tensor> split(long splitSize, long dim) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_SPLIT,
                    LibTorchBinding.TENSOR_SPLIT_DESC);
            MemorySegment countPtr = opArena.allocate(ValueLayout.JAVA_LONG);
            MemorySegment ptrArray = (MemorySegment) fn.invoke(nativeHandle, splitSize, dim, countPtr);
            long count = countPtr.get(ValueLayout.JAVA_LONG, 0);
            MemorySegment sizedPtrArray = ptrArray.reinterpret(count * ValueLayout.ADDRESS.byteSize());

            java.util.List<Tensor> result = new java.util.ArrayList<>((int) count);
            for (long i = 0; i < count; i++) {
                MemorySegment tensorPtr = sizedPtrArray.getAtIndex(ValueLayout.ADDRESS, i);
                result.add(new TorchTensor(tensorPtr, Arena.ofConfined()));
            }

            // Free the array allocated by C++
            MethodHandle freeFn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_FREE_ARRAY,
                    LibTorchBinding.TENSOR_FREE_ARRAY_DESC);
            freeFn.invoke(ptrArray, count);

            return result;
        } catch (Throwable t) {
            throw new RuntimeException("Failed to split tensor", t);
        } finally {
            opArena.close();
        }
    }

    // ── Autograd ──────────────────────────────────────────────────────

    /**
     * Computes gradients via backward pass.
     */
    public void backward() {
        checkClosed();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_BACKWARD,
                    LibTorchBinding.TENSOR_BACKWARD_DESC);
            fn.invoke(nativeHandle);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to compute backward", t);
        }
    }

    /**
     * Returns the gradient tensor.
     *
     * @return the gradient tensor
     */
    public TorchTensor grad() {
        checkClosed();
        return invokeUnaryOp(LibTorchBinding.TENSOR_GRAD);
    }

    /**
     * Enables gradient tracking.
     *
     * @param requiresGrad whether to require gradients
     * @return a new tensor with updated gradient setting
     */
    public TorchTensor requiresGrad(boolean requiresGrad) {
        checkClosed();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_REQUIRES_GRAD,
                    LibTorchBinding.TENSOR_REQUIRES_GRAD_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, requiresGrad);
            return new TorchTensor(result, Arena.ofConfined());
        } catch (Throwable t) {
            throw new RuntimeException("Failed to set requires_grad", t);
        }
    }

    /**
     * Copies elements from another tensor into this tensor (in-place).
     *
     * @param other the source tensor
     */
    public void copy_(TorchTensor other) {
        checkClosed();
        other.checkClosed();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_COPY_INPLACE,
                    LibTorchBinding.TENSOR_COPY_INPLACE_DESC);
            fn.invoke(nativeHandle, other.nativeHandle());
        } catch (Throwable t) {
            throw new RuntimeException("Failed to copy_ tensor", t);
        }
    }

    // ── Device operations ─────────────────────────────────────────────

    /**
     * Moves tensor to a different device.
     *
     * @param device the target device
     * @return a new tensor on the target device
     */
    public TorchTensor to(Device device) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_TO_DEVICE,
                    LibTorchBinding.TENSOR_TO_DEVICE_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, device.type().code(), device.index());
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to move tensor to device " + device, t);
        }
    }

    /**
     * Get the device this tensor is allocated on.
     * Returns CPU by default since device detection from LibTorch may not be
     * exposed.
     * 
     * @return the Device this tensor is on
     */
    public Device getDevice() {
        Device.Type type = deviceType();
        if (type == Device.Type.CPU) {
            return Device.CPU;
        }
        return new Device(type, 0);
    }

    /**
     * Casts tensor to a different scalar type.
     *
     * @param dtype the target scalar type
     * @return a new tensor with the new dtype
     */
    public TorchTensor to(ScalarType dtype) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_TO_DTYPE,
                    LibTorchBinding.TENSOR_TO_DTYPE_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, dtype.code());
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to cast tensor to dtype " + dtype, t);
        }
    }

    /**
     * Clones this tensor (deep copy).
     *
     * @return a new tensor with copied data
     */
    public TorchTensor clone_() {
        checkClosed();
        return invokeUnaryOp(LibTorchBinding.TENSOR_CLONE);
    }

    // ── Internal helpers ──────────────────────────────────────────────

    private TorchTensor invokeBinaryOp(String fnName, TorchTensor other) {
        checkClosed();
        other.checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(fnName, LibTorchBinding.TENSOR_BINARY_OP_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, other.nativeHandle);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to execute " + fnName, t);
        }
    }

    private TorchTensor invokeUnaryOp(String fnName) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(fnName, LibTorchBinding.TENSOR_UNARY_OP_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to execute " + fnName, t);
        }
    }

    private long invokeUnaryLong(String fnName, java.lang.foreign.FunctionDescriptor desc) {
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(fnName, desc);
            return (long) fn.invoke(nativeHandle);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to execute " + fnName, t);
        }
    }

    /**
     * Checks if this tensor has been closed.
     *
     * @return true if closed, false otherwise
     */
    public boolean isClosed() {
        return closed;
    }

    private void checkClosed() {
        if (closed) {
            throw new IllegalStateException("Tensor has been closed and cannot be used");
        }
    }

    /**
     * Returns the underlying native handle.
     * <p>
     * For internal use by LibTorch bindings only.
     * </p>
     *
     * @return the native memory segment
     */
    public MemorySegment nativeHandle() {
        checkClosed();
        return nativeHandle;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            LIVE_COUNT.decrementAndGet();
            // Cancel the cleaner — we are cleaning up explicitly
            cleanable.clean();
        }
    }

    @Override
    public String toString() {
        if (closed)
            return "TorchTensor(closed)";
        try {
            return String.format("TorchTensor(shape=%s, dtype=%s)", Arrays.toString(shape()), dtype());
        } catch (Exception e) {
            return "TorchTensor(native)";
        }
    }

    // ── Cleaner safety net ────────────────────────────────────────────

    /**
     * Static cleanup action registered with the {@link Cleaner}.
     * <p>
     * This must be a <em>static</em> class (or a record) — it must NOT capture a
     * reference to the {@code TorchTensor} instance, otherwise the cleaner will
     * never
     * run because the instance would be reachable from the cleaning action.
     * </p>
     */
    private static final class TensorCleaner implements Runnable {
        private final MemorySegment handle;
        private final Arena arena;
        private final Exception allocationSite;

        TensorCleaner(MemorySegment handle, Arena arena, Exception allocationSite) {
            this.handle = handle;
            this.arena = arena;
            this.allocationSite = allocationSite;
        }

        @Override
        public void run() {
            // If we get here via Cleaner (not via explicit close()), it means
            // the Tensor was GC'd without being closed — a resource leak.
            LIVE_COUNT.decrementAndGet();

            if (allocationSite != null) {
                LEAK_LOGGER.log(System.Logger.Level.WARNING,
                        "TorchTensor was not closed! Allocated at:", allocationSite);
            } else {
                LEAK_LOGGER.log(System.Logger.Level.WARNING,
                        "TorchTensor was not closed! Enable -Dgollek.tensor.leak-detection=true for stack traces.");
            }

            // Best-effort native cleanup
            try {
                LibTorchBinding.getInstance()
                        .bindOptional(LibTorchBinding.TENSOR_FREE, LibTorchBinding.TENSOR_FREE_DESC)
                        .ifPresent(fn -> {
                            try {
                                fn.invoke(handle);
                            } catch (Throwable ignored) {
                            }
                        });
            } catch (Throwable ignored) {
            }

            try {
                arena.close();
            } catch (Throwable ignored) {
            }
        }
    }

    private static final class CleanerHolder {
        private static final Cleaner CLEANER = Cleaner.create();

        private CleanerHolder() {
        }
    }
}
