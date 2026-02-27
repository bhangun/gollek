package tech.kayys.gollek.inference.libtorch.core;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * High-level tensor abstraction wrapping native LibTorch tensors via FFM.
 * <p>
 * Each tensor owns a confined {@link Arena} for its native memory. When closed,
 * the arena is released and the underlying native tensor is freed.
 * <p>
 * A {@link Cleaner}-based safety net ensures native memory is freed even if
 * the caller forgets to call {@link #close()}. A warning is logged when this
 * happens to help track resource leaks.
 * <p>
 * Not thread-safe — each tensor should be used from a single thread.
 */
public class Tensor implements AutoCloseable {

    private static final System.Logger LEAK_LOGGER = System.getLogger("tech.kayys.gollek.tensor.leak");
    private static final Cleaner CLEANER = Cleaner.create();
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
     * Create a tensor wrapper around an existing native handle.
     *
     * @param nativeHandle pointer to the native at::Tensor
     * @param arena        arena managing the native memory
     */
    public Tensor(MemorySegment nativeHandle, Arena arena) {
        this.nativeHandle = Objects.requireNonNull(nativeHandle, "nativeHandle must not be null");
        this.arena = Objects.requireNonNull(arena, "arena must not be null");
        this.allocationSite = LEAK_DETECTION ? new Exception("Tensor allocated here") : null;
        this.cleanable = CLEANER.register(this, new TensorCleaner(nativeHandle, arena, allocationSite));
        LIVE_COUNT.incrementAndGet();
    }

    /**
     * Get the number of currently live (unclosed) tensors.
     * Useful for diagnostics and leak tracking.
     */
    public static long liveCount() {
        return LIVE_COUNT.get();
    }

    // ── Factory methods ───────────────────────────────────────────────

    /**
     * Create an uninitialized tensor with the given shape and type.
     */
    public static Tensor empty(long[] shape, ScalarType dtype, Device device) {
        return createTensor(LibTorchBinding.TENSOR_EMPTY, LibTorchBinding.TENSOR_EMPTY_DESC, shape, dtype, device);
    }

    /**
     * Create a tensor filled with zeros.
     */
    public static Tensor zeros(long[] shape, ScalarType dtype, Device device) {
        return createTensor(LibTorchBinding.TENSOR_ZEROS, LibTorchBinding.TENSOR_ZEROS_DESC, shape, dtype, device);
    }

    /**
     * Create a tensor filled with ones.
     */
    public static Tensor ones(long[] shape, ScalarType dtype, Device device) {
        return createTensor(LibTorchBinding.TENSOR_ONES, LibTorchBinding.TENSOR_ONES_DESC, shape, dtype, device);
    }

    /**
     * Create a tensor with random values from a normal distribution.
     */
    public static Tensor randn(long[] shape, ScalarType dtype, Device device) {
        return createTensor(LibTorchBinding.TENSOR_RANDN, LibTorchBinding.TENSOR_RANDN_DESC, shape, dtype, device);
    }

    /**
     * Create a tensor with random values from a uniform distribution [0, 1).
     */
    public static Tensor rand(long[] shape, ScalarType dtype, Device device) {
        return createTensor(LibTorchBinding.TENSOR_RAND, LibTorchBinding.TENSOR_RAND_DESC, shape, dtype, device);
    }

    /** Convenience: create a float tensor on CPU. */
    public static Tensor zeros(long... shape) {
        return zeros(shape, ScalarType.FLOAT, Device.CPU);
    }

    /** Convenience: create a float tensor on CPU. */
    public static Tensor ones(long... shape) {
        return ones(shape, ScalarType.FLOAT, Device.CPU);
    }

    /** Convenience: create a float tensor on CPU. */
    public static Tensor randn(long... shape) {
        return randn(shape, ScalarType.FLOAT, Device.CPU);
    }

    /**
     * Create a tensor from a float array.
     */
    public static Tensor fromFloatArray(float[] data, long[] shape) {
        Arena arena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(LibTorchBinding.TENSOR_FROM_BLOB, LibTorchBinding.TENSOR_FROM_BLOB_DESC);

            MemorySegment dataSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, data);
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment result = (MemorySegment) fn.invoke(dataSegment, shapeSegment, (long) shape.length,
                    ScalarType.FLOAT.code());
            return new Tensor(result, arena);
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to create tensor from float array", t);
        }
    }

    /**
     * Create a tensor from a double array.
     */
    public static Tensor fromDoubleArray(double[] data, long[] shape) {
        Arena arena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(LibTorchBinding.TENSOR_FROM_BLOB, LibTorchBinding.TENSOR_FROM_BLOB_DESC);

            MemorySegment dataSegment = arena.allocateFrom(ValueLayout.JAVA_DOUBLE, data);
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment result = (MemorySegment) fn.invoke(dataSegment, shapeSegment, (long) shape.length,
                    ScalarType.DOUBLE.code());
            return new Tensor(result, arena);
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to create tensor from double array", t);
        }
    }

    /**
     * Create a tensor from a long array (e.g. for index tensors).
     */
    public static Tensor fromLongArray(long[] data, long[] shape) {
        Arena arena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(LibTorchBinding.TENSOR_FROM_BLOB, LibTorchBinding.TENSOR_FROM_BLOB_DESC);

            MemorySegment dataSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, data);
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment result = (MemorySegment) fn.invoke(dataSegment, shapeSegment, (long) shape.length,
                    ScalarType.LONG.code());
            return new Tensor(result, arena);
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to create tensor from long array", t);
        }
    }

    /**
     * Concatenate a list of tensors along the given dimension.
     * Used by the continuous batching manager to merge individual inputs into a
     * batch.
     *
     * @param tensors list of tensors to concatenate
     * @param dim     dimension along which to concatenate (typically 0 for
     *                batching)
     * @return a new tensor representing the concatenated result
     */
    public static Tensor cat(java.util.List<Tensor> tensors, long dim) {
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
            return new Tensor(result, arena);
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to concatenate tensors", t);
        }
    }

    /**
     * Select elements along a dimension using an index tensor.
     * Used by the continuous batching manager to slice batched output back to
     * individual results.
     *
     * @param dim   dimension along which to select
     * @param index 1-D index tensor
     * @return a new tensor with the selected elements
     */
    public Tensor indexSelect(long dim, Tensor index) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(LibTorchBinding.TENSOR_INDEX_SELECT,
                    LibTorchBinding.TENSOR_INDEX_SELECT_DESC);

            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, dim, index.nativeHandle());
            return new Tensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("index_select failed", t);
        }
    }

    /**
     * Returns the indices of the maximum value of all elements in the input tensor.
     *
     * @param dim the dimension to reduce. If null, the argmax of the flattened
     *            input is returned.
     * @return the result tensor
     */

    /**
     * Returns the value of this tensor as a standard Java long.
     * This only works for tensors with one element.
     *
     * @return the long value
     */
    public long itemLong() {
        checkClosed();
        if (numel() != 1) {
            throw new IllegalStateException("Tensor must have exactly one element to call itemLong()");
        }

        // Ideally we should check device and move to CPU if needed
        // For now, assuming CPU or that we can access unified memory if available
        // But to be safe, we should probably ensure it's on CPU
        // TODO: Enforce to(CPU) check

        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle dataPtrFn = binding.bind(LibTorchBinding.TENSOR_DATA_PTR,
                    LibTorchBinding.TENSOR_DATA_PTR_DESC);

            MemorySegment ptr = (MemorySegment) dataPtrFn.invoke(nativeHandle);
            // Reinterpret as long buffer of size 8
            MemorySegment data = ptr.reinterpret(8);

            // Check dtype
            ScalarType currentDtype = dtype();
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

    private static Tensor createTensor(String fnName, java.lang.foreign.FunctionDescriptor desc,
            long[] shape, ScalarType dtype, Device device) {
        Arena arena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(fnName, desc);

            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment result = (MemorySegment) fn.invoke(shapeSegment, (long) shape.length, dtype.code(),
                    device.type().code());
            return new Tensor(result, arena);
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to create tensor via " + fnName, t);
        }
    }

    // ── Properties ────────────────────────────────────────────────────

    /** Get the number of dimensions. */
    public long dim() {
        checkClosed();
        return invokeUnaryLong(LibTorchBinding.TENSOR_DIM, LibTorchBinding.TENSOR_DIM_DESC);
    }

    /** Get the total number of elements. */
    public long numel() {
        checkClosed();
        return invokeUnaryLong(LibTorchBinding.TENSOR_NUMEL, LibTorchBinding.TENSOR_NUMEL_DESC);
    }

    /** Get the size along a specific dimension. */
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

    /** Get the shape as an array. */
    public long[] shape() {
        checkClosed();
        int ndim = (int) dim();
        long[] result = new long[ndim];
        for (int i = 0; i < ndim; i++) {
            result[i] = size(i);
        }
        return result;
    }

    /** Get the scalar type (dtype). */
    public ScalarType dtype() {
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

    /** Get the device type. */
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

    /** Get the raw data pointer. */
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

    /** Copy tensor data to a float array. */
    public float[] toFloatArray() {
        checkClosed();
        long n = numel();
        MemorySegment data = dataPtr().reinterpret(n * ValueLayout.JAVA_FLOAT.byteSize());
        return data.toArray(ValueLayout.JAVA_FLOAT);
    }

    /** Copy tensor data to a double array. */
    public double[] toDoubleArray() {
        checkClosed();
        long n = numel();
        MemorySegment data = dataPtr().reinterpret(n * ValueLayout.JAVA_DOUBLE.byteSize());
        return data.toArray(ValueLayout.JAVA_DOUBLE);
    }

    /** Copy tensor data to a long array. */
    public long[] toLongArray() {
        checkClosed();
        long n = numel();
        MemorySegment data = dataPtr().reinterpret(n * ValueLayout.JAVA_LONG.byteSize());
        return data.toArray(ValueLayout.JAVA_LONG);
    }

    // ── Binary operations ─────────────────────────────────────────────

    /** Element-wise addition. */
    public Tensor add(Tensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_ADD, other);
    }

    /** Element-wise subtraction. */
    public Tensor sub(Tensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_SUB, other);
    }

    /** Element-wise multiplication. */
    public Tensor mul(Tensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_MUL, other);
    }

    /** Element-wise division. */
    public Tensor div(Tensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_DIV, other);
    }

    /** Matrix multiplication. */
    public Tensor matmul(Tensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_MATMUL, other);
    }

    // ── Comparison operations ─────────────────────────────────────────

    /** Element-wise equality. */
    public Tensor eq(Tensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_EQ, other);
    }

    /** Element-wise greater-than. */
    public Tensor gt(Tensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_GT, other);
    }

    /** Element-wise less-than. */
    public Tensor lt(Tensor other) {
        return invokeBinaryOp(LibTorchBinding.TENSOR_LT, other);
    }

    // ── Unary operations ──────────────────────────────────────────────

    /** Negate all elements. */
    public Tensor neg() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_NEG);
    }

    /** Absolute value. */
    public Tensor abs() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_ABS);
    }

    /** Square root. */
    public Tensor sqrt() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_SQRT);
    }

    /** Natural logarithm. */
    public Tensor log() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_LOG);
    }

    /** Exponential. */
    public Tensor exp() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_EXP);
    }

    // ── Reduction operations ──────────────────────────────────────────

    /** Sum of all elements. */
    public Tensor sum() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_SUM);
    }

    /** Mean of all elements. */
    public Tensor mean() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_MEAN);
    }

    /** Maximum element. */
    public Tensor max() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_MAX);
    }

    /** Minimum element. */
    public Tensor min() {
        return invokeUnaryOp(LibTorchBinding.TENSOR_MIN);
    }

    /** Index of the maximum element along a dimension. */
    public Tensor argmax(long dim) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_ARGMAX,
                    LibTorchBinding.TENSOR_ARGMAX_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, dim);
            return new Tensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to compute argmax", t);
        }
    }

    // ── Shape operations ──────────────────────────────────────────────

    /** Reshape to a new shape. */
    public Tensor reshape(long... newShape) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            LibTorchBinding binding = LibTorchBinding.getInstance();
            MethodHandle fn = binding.bind(LibTorchBinding.TENSOR_RESHAPE, LibTorchBinding.TENSOR_RESHAPE_DESC);
            MemorySegment shapeSegment = opArena.allocateFrom(ValueLayout.JAVA_LONG, newShape);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, shapeSegment, (long) newShape.length);
            return new Tensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to reshape tensor", t);
        }
    }

    /** Transpose two dimensions. */
    public Tensor transpose(long dim0, long dim1) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_TRANSPOSE,
                    LibTorchBinding.TENSOR_TRANSPOSE_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, dim0, dim1);
            return new Tensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to transpose tensor", t);
        }
    }

    // ── Autograd ──────────────────────────────────────────────────────

    /** Compute gradients via backward pass. */
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

    /** Get the gradient tensor. */
    public Tensor grad() {
        checkClosed();
        return invokeUnaryOp(LibTorchBinding.TENSOR_GRAD);
    }

    /** Enable gradient tracking. */
    public Tensor requiresGrad(boolean requiresGrad) {
        checkClosed();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_REQUIRES_GRAD,
                    LibTorchBinding.TENSOR_REQUIRES_GRAD_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, requiresGrad);
            return new Tensor(result, Arena.ofConfined());
        } catch (Throwable t) {
            throw new RuntimeException("Failed to set requires_grad", t);
        }
    }

    // ── Device operations ─────────────────────────────────────────────

    /** Move tensor to a different device. */
    public Tensor to(Device device) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(LibTorchBinding.TENSOR_TO_DEVICE,
                    LibTorchBinding.TENSOR_TO_DEVICE_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, device.type().code(), device.index());
            return new Tensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to move tensor to device " + device, t);
        }
    }

    /** Clone this tensor (deep copy). */
    public Tensor clone_() {
        checkClosed();
        return invokeUnaryOp(LibTorchBinding.TENSOR_CLONE);
    }

    // ── Internal helpers ──────────────────────────────────────────────

    private Tensor invokeBinaryOp(String fnName, Tensor other) {
        checkClosed();
        other.checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(fnName, LibTorchBinding.TENSOR_BINARY_OP_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle, other.nativeHandle);
            return new Tensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Failed to execute " + fnName, t);
        }
    }

    private Tensor invokeUnaryOp(String fnName) {
        checkClosed();
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(fnName, LibTorchBinding.TENSOR_UNARY_OP_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(nativeHandle);
            return new Tensor(result, opArena);
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

    private void checkClosed() {
        if (closed) {
            throw new IllegalStateException("Tensor has been closed and cannot be used");
        }
    }

    /** Get the underlying native handle. */
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
            return "Tensor(closed)";
        try {
            return String.format("Tensor(shape=%s, dtype=%s)", Arrays.toString(shape()), dtype());
        } catch (Exception e) {
            return "Tensor(native)";
        }
    }

    // ── Cleaner safety net ────────────────────────────────────────────

    /**
     * Static cleanup action registered with the {@link Cleaner}.
     * <p>
     * This must be a <em>static</em> class (or a record) — it must NOT capture a
     * reference to the {@code Tensor} instance, otherwise the cleaner will never
     * run because the instance would be reachable from the cleaning action.
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
                        "Tensor was not closed! Allocated at:", allocationSite);
            } else {
                LEAK_LOGGER.log(System.Logger.Level.WARNING,
                        "Tensor was not closed! Enable -Dgollek.tensor.leak-detection=true for stack traces.");
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
}
