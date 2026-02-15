package io.github.pytorch.core;

import io.github.pytorch.ffm.LibTorchFFM;

import java.lang.foreign.*;
import java.util.Arrays;

/**
 * High-level Tensor class wrapping LibTorch tensors using FFM API
 */
public class Tensor implements AutoCloseable {
    
    private final MemorySegment nativeHandle;
    private final Arena arena;
    private boolean closed = false;
    
    /**
     * Private constructor - use factory methods
     */
    private Tensor(MemorySegment handle, Arena arena) {
        this.nativeHandle = handle;
        this.arena = arena;
    }
    
    /**
     * Get the native memory handle
     */
    public MemorySegment handle() {
        checkClosed();
        return nativeHandle;
    }
    
    // ============================================================================
    // Factory Methods - Tensor Creation
    // ============================================================================
    
    /**
     * Create an empty tensor with the given shape
     */
    public static Tensor empty(long[] shape) {
        return empty(shape, ScalarType.FLOAT, Device.CPU);
    }
    
    public static Tensor empty(long[] shape, ScalarType dtype, Device device) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment options = createTensorOptions(dtype, device, arena);
            
            MemorySegment handle = (MemorySegment) LibTorchFFM.AT_EMPTY.invoke(
                shapeSegment, shape.length, options
            );
            
            return new Tensor(handle, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create empty tensor", e);
        }
    }
    
    /**
     * Create a tensor filled with zeros
     */
    public static Tensor zeros(long[] shape) {
        return zeros(shape, ScalarType.FLOAT, Device.CPU);
    }
    
    public static Tensor zeros(long[] shape, ScalarType dtype, Device device) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment options = createTensorOptions(dtype, device, arena);
            
            MemorySegment handle = (MemorySegment) LibTorchFFM.AT_ZEROS.invoke(
                shapeSegment, shape.length, options
            );
            
            return new Tensor(handle, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create zeros tensor", e);
        }
    }
    
    /**
     * Create a tensor filled with ones
     */
    public static Tensor ones(long[] shape) {
        return ones(shape, ScalarType.FLOAT, Device.CPU);
    }
    
    public static Tensor ones(long[] shape, ScalarType dtype, Device device) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment options = createTensorOptions(dtype, device, arena);
            
            MemorySegment handle = (MemorySegment) LibTorchFFM.AT_ONES.invoke(
                shapeSegment, shape.length, options
            );
            
            return new Tensor(handle, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create ones tensor", e);
        }
    }
    
    /**
     * Create a tensor with random values from normal distribution
     */
    public static Tensor randn(long[] shape) {
        return randn(shape, ScalarType.FLOAT, Device.CPU);
    }
    
    public static Tensor randn(long[] shape, ScalarType dtype, Device device) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment options = createTensorOptions(dtype, device, arena);
            
            MemorySegment handle = (MemorySegment) LibTorchFFM.AT_RANDN.invoke(
                shapeSegment, shape.length, options
            );
            
            return new Tensor(handle, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create randn tensor", e);
        }
    }
    
    /**
     * Create a tensor with random values from uniform distribution [0, 1)
     */
    public static Tensor rand(long[] shape) {
        return rand(shape, ScalarType.FLOAT, Device.CPU);
    }
    
    public static Tensor rand(long[] shape, ScalarType dtype, Device device) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment options = createTensorOptions(dtype, device, arena);
            
            MemorySegment handle = (MemorySegment) LibTorchFFM.AT_RAND.invoke(
                shapeSegment, shape.length, options
            );
            
            return new Tensor(handle, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create rand tensor", e);
        }
    }
    
    /**
     * Create a tensor from a range of values
     */
    public static Tensor arange(double start, double end, double step) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment options = createTensorOptions(ScalarType.FLOAT, Device.CPU, arena);
            
            MemorySegment handle = (MemorySegment) LibTorchFFM.AT_ARANGE.invoke(
                start, end, step, options
            );
            
            return new Tensor(handle, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create arange tensor", e);
        }
    }
    
    /**
     * Create a tensor with linearly spaced values
     */
    public static Tensor linspace(double start, double end, long steps) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment options = createTensorOptions(ScalarType.FLOAT, Device.CPU, arena);
            
            MemorySegment handle = (MemorySegment) LibTorchFFM.AT_LINSPACE.invoke(
                start, end, steps, options
            );
            
            return new Tensor(handle, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create linspace tensor", e);
        }
    }
    
    /**
     * Create a tensor from Java array (float)
     */
    public static Tensor fromArray(float[] data, long[] shape) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment dataSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, data);
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment options = createTensorOptions(ScalarType.FLOAT, Device.CPU, arena);
            
            MemorySegment handle = (MemorySegment) LibTorchFFM.AT_FROM_BLOB.invoke(
                dataSegment, shapeSegment, shape.length, options
            );
            
            return new Tensor(handle, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create tensor from array", e);
        }
    }
    
    /**
     * Create a tensor from Java array (double)
     */
    public static Tensor fromArray(double[] data, long[] shape) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment dataSegment = arena.allocateFrom(ValueLayout.JAVA_DOUBLE, data);
            MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment options = createTensorOptions(ScalarType.DOUBLE, Device.CPU, arena);
            
            MemorySegment handle = (MemorySegment) LibTorchFFM.AT_FROM_BLOB.invoke(
                dataSegment, shapeSegment, shape.length, options
            );
            
            return new Tensor(handle, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create tensor from array", e);
        }
    }
    
    // ============================================================================
    // Tensor Operations
    // ============================================================================
    
    /**
     * Add another tensor or scalar
     */
    public Tensor add(Tensor other) {
        return add(other, 1.0);
    }
    
    public Tensor add(Tensor other, double alpha) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_ADD.invoke(
                nativeHandle, other.nativeHandle, alpha
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to add tensors", e);
        }
    }
    
    public Tensor add(double scalar) {
        return add(Tensor.fromArray(new double[]{scalar}, new long[]{1}));
    }
    
    /**
     * Subtract another tensor or scalar
     */
    public Tensor sub(Tensor other) {
        return sub(other, 1.0);
    }
    
    public Tensor sub(Tensor other, double alpha) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_SUB.invoke(
                nativeHandle, other.nativeHandle, alpha
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to subtract tensors", e);
        }
    }
    
    public Tensor sub(double scalar) {
        return sub(Tensor.fromArray(new double[]{scalar}, new long[]{1}));
    }
    
    /**
     * Multiply by another tensor or scalar
     */
    public Tensor mul(Tensor other) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_MUL.invoke(
                nativeHandle, other.nativeHandle
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to multiply tensors", e);
        }
    }
    
    public Tensor mul(double scalar) {
        return mul(Tensor.fromArray(new double[]{scalar}, new long[]{1}));
    }
    
    /**
     * Divide by another tensor or scalar
     */
    public Tensor div(Tensor other) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_DIV.invoke(
                nativeHandle, other.nativeHandle
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to divide tensors", e);
        }
    }
    
    public Tensor div(double scalar) {
        return div(Tensor.fromArray(new double[]{scalar}, new long[]{1}));
    }
    
    /**
     * Matrix multiplication
     */
    public Tensor matmul(Tensor other) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_MATMUL.invoke(
                nativeHandle, other.nativeHandle
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to perform matmul", e);
        }
    }
    
    /**
     * Matrix-matrix product (2D tensors only)
     */
    public Tensor mm(Tensor other) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_MM.invoke(
                nativeHandle, other.nativeHandle
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to perform mm", e);
        }
    }
    
    /**
     * Batch matrix-matrix product
     */
    public Tensor bmm(Tensor other) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_BMM.invoke(
                nativeHandle, other.nativeHandle
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to perform bmm", e);
        }
    }
    
    /**
     * Reshape tensor to new shape
     */
    public Tensor reshape(long[] shape) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment shapeSegment = resultArena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_RESHAPE.invoke(
                nativeHandle, shapeSegment, shape.length
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to reshape tensor", e);
        }
    }
    
    /**
     * View tensor as new shape (must be compatible)
     */
    public Tensor view(long[] shape) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment shapeSegment = resultArena.allocateFrom(ValueLayout.JAVA_LONG, shape);
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_VIEW.invoke(
                nativeHandle, shapeSegment, shape.length
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to view tensor", e);
        }
    }
    
    /**
     * Transpose two dimensions
     */
    public Tensor transpose(int dim0, int dim1) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_TRANSPOSE.invoke(
                nativeHandle, dim0, dim1
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to transpose tensor", e);
        }
    }
    
    /**
     * Permute dimensions
     */
    public Tensor permute(int[] dims) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment dimsSegment = resultArena.allocateFrom(ValueLayout.JAVA_INT, dims);
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_PERMUTE.invoke(
                nativeHandle, dimsSegment, dims.length
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to permute tensor", e);
        }
    }
    
    /**
     * Remove dimensions of size 1
     */
    public Tensor squeeze() {
        return squeeze(-1);
    }
    
    public Tensor squeeze(int dim) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_SQUEEZE.invoke(
                nativeHandle, dim
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to squeeze tensor", e);
        }
    }
    
    /**
     * Add a dimension of size 1
     */
    public Tensor unsqueeze(int dim) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_UNSQUEEZE.invoke(
                nativeHandle, dim
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to unsqueeze tensor", e);
        }
    }
    
    /**
     * Sum all elements
     */
    public Tensor sum() {
        return sum(null, false);
    }
    
    public Tensor sum(int[] dims, boolean keepdim) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment dimsSegment = dims != null ? 
                resultArena.allocateFrom(ValueLayout.JAVA_INT, dims) : MemorySegment.NULL;
            int dimCount = dims != null ? dims.length : 0;
            
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_SUM.invoke(
                nativeHandle, dimsSegment, dimCount, keepdim
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to sum tensor", e);
        }
    }
    
    /**
     * Mean of all elements
     */
    public Tensor mean() {
        return mean(null, false);
    }
    
    public Tensor mean(int[] dims, boolean keepdim) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment dimsSegment = dims != null ? 
                resultArena.allocateFrom(ValueLayout.JAVA_INT, dims) : MemorySegment.NULL;
            int dimCount = dims != null ? dims.length : 0;
            
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_MEAN.invoke(
                nativeHandle, dimsSegment, dimCount, keepdim
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to compute mean", e);
        }
    }
    
    /**
     * Maximum value
     */
    public Tensor max() {
        return max(-1, false);
    }
    
    public Tensor max(int dim, boolean keepdim) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_MAX.invoke(
                nativeHandle, dim, keepdim
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to compute max", e);
        }
    }
    
    /**
     * Minimum value
     */
    public Tensor min() {
        return min(-1, false);
    }
    
    public Tensor min(int dim, boolean keepdim) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_MIN.invoke(
                nativeHandle, dim, keepdim
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to compute min", e);
        }
    }
    
    // ============================================================================
    // Tensor Properties
    // ============================================================================
    
    /**
     * Get tensor shape
     */
    public long[] shape() {
        checkClosed();
        try {
            long ndim = (long) LibTorchFFM.TENSOR_DIM.invoke(nativeHandle);
            MemorySegment sizesPtr = (MemorySegment) LibTorchFFM.TENSOR_SIZES.invoke(nativeHandle);
            
            long[] shape = new long[(int) ndim];
            for (int i = 0; i < ndim; i++) {
                shape[i] = sizesPtr.getAtIndex(ValueLayout.JAVA_LONG, i);
            }
            return shape;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor shape", e);
        }
    }
    
    /**
     * Get size of a specific dimension
     */
    public long size(int dim) {
        checkClosed();
        try {
            return (long) LibTorchFFM.TENSOR_SIZE.invoke(nativeHandle, dim);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor size", e);
        }
    }
    
    /**
     * Get number of dimensions
     */
    public long ndim() {
        checkClosed();
        try {
            return (long) LibTorchFFM.TENSOR_DIM.invoke(nativeHandle);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor ndim", e);
        }
    }
    
    /**
     * Get total number of elements
     */
    public long numel() {
        checkClosed();
        try {
            return (long) LibTorchFFM.TENSOR_NUMEL.invoke(nativeHandle);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor numel", e);
        }
    }
    
    /**
     * Get scalar type
     */
    public ScalarType dtype() {
        checkClosed();
        try {
            int typeCode = (int) LibTorchFFM.TENSOR_SCALAR_TYPE.invoke(nativeHandle);
            return ScalarType.fromCode(typeCode);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor dtype", e);
        }
    }
    
    /**
     * Get device
     */
    public Device device() {
        checkClosed();
        try {
            MemorySegment devicePtr = (MemorySegment) LibTorchFFM.TENSOR_DEVICE.invoke(nativeHandle);
            // Parse device from pointer
            return Device.CPU; // TODO: Parse properly
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor device", e);
        }
    }
    
    /**
     * Check if tensor requires gradient
     */
    public boolean requiresGrad() {
        checkClosed();
        try {
            return (boolean) LibTorchFFM.TENSOR_REQUIRES_GRAD.invoke(nativeHandle);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to check requires_grad", e);
        }
    }
    
    /**
     * Get raw data pointer (use with caution!)
     */
    public MemorySegment dataPtr() {
        checkClosed();
        try {
            return (MemorySegment) LibTorchFFM.TENSOR_DATA_PTR.invoke(nativeHandle);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get data pointer", e);
        }
    }
    
    // ============================================================================
    // Device Management
    // ============================================================================
    
    /**
     * Move tensor to device
     */
    public Tensor to(Device device) {
        return to(device, null, false, false);
    }
    
    public Tensor to(Device device, ScalarType dtype, boolean nonBlocking, boolean copy) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment deviceSegment = createDeviceSegment(device, resultArena);
            int dtypeCode = dtype != null ? dtype.code() : -1;
            
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_TO.invoke(
                nativeHandle, deviceSegment, dtypeCode, nonBlocking, copy
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to move tensor to device", e);
        }
    }
    
    /**
     * Move tensor to CUDA
     */
    public Tensor cuda() {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_CUDA.invoke(nativeHandle);
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to move tensor to CUDA", e);
        }
    }
    
    /**
     * Move tensor to CPU
     */
    public Tensor cpu() {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_CPU.invoke(nativeHandle);
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to move tensor to CPU", e);
        }
    }
    
    /**
     * Clone tensor
     */
    public Tensor clone() {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_CLONE.invoke(nativeHandle);
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to clone tensor", e);
        }
    }
    
    /**
     * Detach tensor from computation graph
     */
    public Tensor detach() {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_DETACH.invoke(nativeHandle);
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to detach tensor", e);
        }
    }
    
    // ============================================================================
    // Autograd
    // ============================================================================
    
    /**
     * Compute gradients
     */
    public void backward() {
        backward(null, false, false);
    }
    
    public void backward(Tensor gradient, boolean retainGraph, boolean createGraph) {
        checkClosed();
        try {
            MemorySegment gradHandle = gradient != null ? gradient.nativeHandle : MemorySegment.NULL;
            LibTorchFFM.TENSOR_BACKWARD.invoke(nativeHandle, gradHandle, retainGraph, createGraph);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to compute backward", e);
        }
    }
    
    /**
     * Get gradient
     */
    public Tensor grad() {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_GRAD.invoke(nativeHandle);
            if (resultHandle.address() == 0) {
                resultArena.close();
                return null;
            }
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to get gradient", e);
        }
    }
    
    /**
     * Zero gradient
     */
    public void zeroGrad() {
        checkClosed();
        try {
            LibTorchFFM.TENSOR_ZERO_GRAD.invoke(nativeHandle);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to zero gradient", e);
        }
    }
    
    /**
     * Set requires_grad flag
     */
    public Tensor requiresGrad(boolean requires) {
        checkClosed();
        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment resultHandle = (MemorySegment) LibTorchFFM.TENSOR_REQUIRES_GRAD_.invoke(
                nativeHandle, requires
            );
            return new Tensor(resultHandle, resultArena);
        } catch (Throwable e) {
            resultArena.close();
            throw new RuntimeException("Failed to set requires_grad", e);
        }
    }
    
    // ============================================================================
    // Utility Methods
    // ============================================================================
    
    private static MemorySegment createTensorOptions(ScalarType dtype, Device device, Arena arena) {
        // Create TensorOptions struct
        // This is a simplified version - actual implementation would need proper struct layout
        MemorySegment options = arena.allocate(32); // Allocate space for options
        options.set(ValueLayout.JAVA_INT, 0, dtype.code());
        options.set(ValueLayout.JAVA_INT, 4, device.type().ordinal());
        return options;
    }
    
    private static MemorySegment createDeviceSegment(Device device, Arena arena) {
        MemorySegment deviceSeg = arena.allocate(8);
        deviceSeg.set(ValueLayout.JAVA_INT, 0, device.type().ordinal());
        deviceSeg.set(ValueLayout.JAVA_INT, 4, device.index());
        return deviceSeg;
    }
    
    private void checkClosed() {
        if (closed) {
            throw new IllegalStateException("Tensor has been closed");
        }
    }


    // Scalar types matching PyTorch ScalarType enum
    public enum ScalarType {
        FLOAT(6),
        DOUBLE(7),
        INT32(3),
        INT64(4),
        UINT8(0),
        INT8(1),
        BOOL(11);
        
        private final int value;
        
        ScalarType(int value) {
            this.value = value;
        }
        
        public int getValue() {
            return value;
        }
    }
    
    // Device types
    public enum Device {
        CPU(0),
        CUDA(1);
        
        private final int value;
        
        Device(int value) {
            this.value = value;
        }
        
        public int getValue() {
            return value;
        }
    }
    
    private Tensor(MemorySegment nativeTensor, Arena arena) {
        this.nativeTensor = nativeTensor;
        this.arena = arena;
    }
    
    /**
     * Creates a tensor filled with zeros.
     */
    public static Tensor zeros(long[] shape, ScalarType dtype) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment shapeSeg = arena.allocate(ValueLayout.JAVA_LONG, shape.length);
            for (int i = 0; i < shape.length; i++) {
                shapeSeg.setAtIndex(ValueLayout.JAVA_LONG, i, shape[i]);
            }
            
            MemorySegment tensorPtr = (MemorySegment) LibTorchFFM.at_zeros.invoke(
                shapeSeg, (long) shape.length, dtype.getValue()
            );
            
            return new Tensor(tensorPtr, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create zeros tensor", e);
        }
    }
    
    /**
     * Creates a tensor filled with ones.
     */
    public static Tensor ones(long[] shape, ScalarType dtype) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment shapeSeg = arena.allocate(ValueLayout.JAVA_LONG, shape.length);
            for (int i = 0; i < shape.length; i++) {
                shapeSeg.setAtIndex(ValueLayout.JAVA_LONG, i, shape[i]);
            }
            
            MemorySegment tensorPtr = (MemorySegment) LibTorchFFM.at_ones.invoke(
                shapeSeg, (long) shape.length, dtype.getValue()
            );
            
            return new Tensor(tensorPtr, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create ones tensor", e);
        }
    }
    
    /**
     * Creates a tensor filled with random values from uniform distribution [0, 1).
     */
    public static Tensor rand(long[] shape, ScalarType dtype) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment shapeSeg = arena.allocate(ValueLayout.JAVA_LONG, shape.length);
            for (int i = 0; i < shape.length; i++) {
                shapeSeg.setAtIndex(ValueLayout.JAVA_LONG, i, shape[i]);
            }
            
            MemorySegment tensorPtr = (MemorySegment) LibTorchFFM.at_rand.invoke(
                shapeSeg, (long) shape.length, dtype.getValue()
            );
            
            return new Tensor(tensorPtr, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create rand tensor", e);
        }
    }
    
    /**
     * Creates a tensor filled with random values from normal distribution N(0, 1).
     */
    public static Tensor randn(long[] shape, ScalarType dtype) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment shapeSeg = arena.allocate(ValueLayout.JAVA_LONG, shape.length);
            for (int i = 0; i < shape.length; i++) {
                shapeSeg.setAtIndex(ValueLayout.JAVA_LONG, i, shape[i]);
            }
            
            MemorySegment tensorPtr = (MemorySegment) LibTorchFFM.at_randn.invoke(
                shapeSeg, (long) shape.length, dtype.getValue()
            );
            
            return new Tensor(tensorPtr, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Failed to create randn tensor", e);
        }
    }
    
    /**
     * Element-wise addition with scalar.
     */
    public Tensor add(double scalar) {
        checkNotClosed();
        try {
            MemorySegment resultPtr = (MemorySegment) LibTorchFFM.at_add.invoke(
                nativeTensor, nativeTensor, scalar
            );
            return new Tensor(resultPtr, Arena.ofConfined());
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add scalar", e);
        }
    }
    
    /**
     * Element-wise addition with another tensor.
     */
    public Tensor add(Tensor other) {
        checkNotClosed();
        other.checkNotClosed();
        try {
            MemorySegment resultPtr = (MemorySegment) LibTorchFFM.at_add.invoke(
                nativeTensor, other.nativeTensor, 1.0
            );
            return new Tensor(resultPtr, Arena.ofConfined());
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add tensors", e);
        }
    }
    
    /**
     * Element-wise multiplication with another tensor.
     */
    public Tensor mul(Tensor other) {
        checkNotClosed();
        other.checkNotClosed();
        try {
            MemorySegment resultPtr = (MemorySegment) LibTorchFFM.at_mul.invoke(
                nativeTensor, other.nativeTensor
            );
            return new Tensor(resultPtr, Arena.ofConfined());
        } catch (Throwable e) {
            throw new RuntimeException("Failed to multiply tensors", e);
        }
    }
    
    /**
     * Matrix multiplication.
     */
    public Tensor matmul(Tensor other) {
        checkNotClosed();
        other.checkNotClosed();
        try {
            MemorySegment resultPtr = (MemorySegment) LibTorchFFM.at_matmul.invoke(
                nativeTensor, other.nativeTensor
            );
            return new Tensor(resultPtr, Arena.ofConfined());
        } catch (Throwable e) {
            throw new RuntimeException("Failed to perform matmul", e);
        }
    }
    
    /**
     * ReLU activation function.
     */
    public Tensor relu() {
        checkNotClosed();
        try {
            MemorySegment resultPtr = (MemorySegment) LibTorchFFM.at_relu.invoke(nativeTensor);
            return new Tensor(resultPtr, Arena.ofConfined());
        } catch (Throwable e) {
            throw new RuntimeException("Failed to apply relu", e);
        }
    }
    
    /**
     * Sigmoid activation function.
     */
    public Tensor sigmoid() {
        checkNotClosed();
        try {
            MemorySegment resultPtr = (MemorySegment) LibTorchFFM.at_sigmoid.invoke(nativeTensor);
            return new Tensor(resultPtr, Arena.ofConfined());
        } catch (Throwable e) {
            throw new RuntimeException("Failed to apply sigmoid", e);
        }
    }
    
    /**
     * Tanh activation function.
     */
    public Tensor tanh() {
        checkNotClosed();
        try {
            MemorySegment resultPtr = (MemorySegment) LibTorchFFM.at_tanh.invoke(nativeTensor);
            return new Tensor(resultPtr, Arena.ofConfined());
        } catch (Throwable e) {
            throw new RuntimeException("Failed to apply tanh", e);
        }
    }
    
    /**
     * Softmax activation function.
     */
    public Tensor softmax(long dim) {
        checkNotClosed();
        try {
            MemorySegment resultPtr = (MemorySegment) LibTorchFFM.at_softmax.invoke(
                nativeTensor, dim, ScalarType.FLOAT.getValue()
            );
            return new Tensor(resultPtr, Arena.ofConfined());
        } catch (Throwable e) {
            throw new RuntimeException("Failed to apply softmax", e);
        }
    }
    
    /**
     * Transpose two dimensions.
     */
    public Tensor transpose(long dim0, long dim1) {
        checkNotClosed();
        try {
            // For simplicity, implementing as permute for 2D case
            // Full implementation would call at_transpose
            if (dim() == 2 && dim0 == 0 && dim1 == 1) {
                // Manual transpose for 2D matrices
                long[] origShape = shape();
                return this; // Placeholder - would need actual transpose implementation
            }
            return this;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to transpose", e);
        }
    }
    
    /**
     * Reshape tensor to new shape.
     */
    public Tensor reshape(long[] newShape) {
        checkNotClosed();
        try {
            // Would call at_reshape or at_view
            // For now, returning this as placeholder
            return this;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to reshape", e);
        }
    }
    
    /**
     * Mean reduction over all elements.
     */
    public Tensor mean() {
        checkNotClosed();
        try {
            // Would call at_mean
            throw new UnsupportedOperationException("Mean operation requires additional FFM bindings");
        } catch (Throwable e) {
            throw new RuntimeException("Failed to compute mean", e);
        }
    }
    
    /**
     * Enable gradient computation for this tensor.
     */
    public Tensor requiresGrad(boolean requires) {
        checkNotClosed();
        try {
            LibTorchFFM.at_set_requires_grad.invoke(nativeTensor, requires);
            return this;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to set requires_grad", e);
        }
    }
    
    /**
     * Compute gradients via backpropagation.
     */
    public void backward() {
        checkNotClosed();
        try {
            LibTorchFFM.at_backward.invoke(nativeTensor);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to backward", e);
        }
    }
    
    /**
     * Get gradient tensor.
     */
    public Tensor grad() {
        checkNotClosed();
        try {
            MemorySegment gradPtr = (MemorySegment) LibTorchFFM.at_grad.invoke(nativeTensor);
            if (gradPtr == null || gradPtr.address() == 0) {
                return null;
            }
            return new Tensor(gradPtr, Arena.ofConfined());
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get gradient", e);
        }
    }
    
    /**
     * Move tensor to specified device.
     */
    public Tensor to(Device device) {
        checkNotClosed();
        try {
            MemorySegment resultPtr = (MemorySegment) LibTorchFFM.at_tensor_to_device.invoke(
                nativeTensor, device.getValue()
            );
            return new Tensor(resultPtr, Arena.ofConfined());
        } catch (Throwable e) {
            throw new RuntimeException("Failed to move tensor to device", e);
        }
    }
    
    /**
     * Get size of specified dimension.
     */
    public long size(int dim) {
        checkNotClosed();
        try {
            return (long) LibTorchFFM.at_tensor_size.invoke(nativeTensor, (long) dim);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor size", e);
        }
    }
    
    /**
     * Get number of dimensions.
     */
    public long dim() {
        checkNotClosed();
        try {
            return (long) LibTorchFFM.at_tensor_dim.invoke(nativeTensor);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get tensor dimensions", e);
        }
    }
    
    /**
     * Get total number of elements.
     */
    public long numel() {
        checkNotClosed();
        try {
            return (long) LibTorchFFM.at_tensor_numel.invoke(nativeTensor);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get number of elements", e);
        }
    }
    
    /**
     * Get shape as array.
     */
    public long[] shape() {
        checkNotClosed();
        long dims = dim();
        long[] shape = new long[(int) dims];
        for (int i = 0; i < dims; i++) {
            shape[i] = size(i);
        }
        return shape;
    }
    
    /**
     * Get raw data pointer.
     */
    public MemorySegment dataPtr() {
        checkNotClosed();
        try {
            return (MemorySegment) LibTorchFFM.at_tensor_data_ptr.invoke(nativeTensor);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get data pointer", e);
        }
    }
    
    /**
     * Check if CUDA is available.
     */
    public static boolean cudaIsAvailable() {
        try {
            return (boolean) LibTorchFFM.at_cuda_is_available.invoke();
        } catch (Throwable e) {
            return false;
        }
    }
    
    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("Tensor has been closed");
        }
    }
    
    @Override
    public void close() {
        if (!closed && nativeTensor != null && nativeTensor.address() != 0) {
            try {
                LibTorchFFM.at_tensor_delete.invoke(nativeTensor);
                arena.close();
            } catch (Throwable e) {
                // Log but don't throw in close()
                System.err.println("Failed to delete tensor: " + e.getMessage());
            }
            closed = true;
        }
    }
    

    
 
    @Override
    public String toString() {
        if (closed) {
            return "Tensor(closed)";
        }
        return String.format("Tensor(shape=%s, dtype=%s, device=%s)", 
            Arrays.toString(shape()), dtype(), device());
    }
}
