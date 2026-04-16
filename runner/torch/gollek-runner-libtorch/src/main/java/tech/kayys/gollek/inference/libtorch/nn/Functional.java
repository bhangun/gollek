package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.runtime.tensor.Tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * Static functional API for neural network operations.
 * Mirrors {@code libtorch::nn::functional}.
 * <p>
 * All methods delegate directly to LibTorch via FFM downcall handles.
 */
public final class Functional {

    private Functional() {
    }

    // ── Activation functions ──────────────────────────────────────────

    /** Rectified Linear Unit: max(0, x). */
    public static TorchTensor relu(TorchTensor input) {
        return invokeUnary(LibTorchBinding.NN_RELU, input);
    }

    /** Gaussian Error Linear Unit. */
    public static TorchTensor gelu(TorchTensor input) {
        return invokeUnary(LibTorchBinding.NN_GELU, input);
    }

    /** Sigmoid activation. */
    public static TorchTensor sigmoid(TorchTensor input) {
        return invokeUnary(LibTorchBinding.NN_SIGMOID, input);
    }

    /** Hyperbolic tangent. */
    public static TorchTensor tanh(TorchTensor input) {
        return invokeUnary(LibTorchBinding.NN_TANH, input);
    }

    /** Softmax along a dimension. */
    public static TorchTensor softmax(TorchTensor input, long dim) {
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(
                    LibTorchBinding.NN_SOFTMAX, LibTorchBinding.NN_SOFTMAX_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(input.nativeHandle(), dim);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("softmax failed", t);
        }
    }

    /** Log-softmax along a dimension. */
    public static TorchTensor logSoftmax(TorchTensor input, long dim) {
        // log(softmax(x)) — using composition
        TorchTensor sm = softmax(input, dim);
        return sm.log();
    }

    // ── Regularization ────────────────────────────────────────────────

    /** Dropout. */
    public static TorchTensor dropout(TorchTensor input, double p, boolean training) {
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(
                    LibTorchBinding.NN_DROPOUT, LibTorchBinding.NN_DROPOUT_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(input.nativeHandle(), p, training);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("dropout failed", t);
        }
    }

    // ── Convolution ───────────────────────────────────────────────────

    /**
     * 2D convolution.
     *
     * @param input    input tensor [N, C_in, H, W]
     * @param weight   weight tensor [C_out, C_in/groups, kH, kW]
     * @param bias     optional bias tensor [C_out], may be null
     * @param stride   convolution stride
     * @param padding  zero-padding
     * @param dilation dilation factor
     * @param groups   number of blocked connections
     * @return output tensor
     */
    public static TorchTensor conv2d(TorchTensor input, TorchTensor weight, TorchTensor bias,
            long stride, long padding, long dilation, long groups) {
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(
                    LibTorchBinding.NN_CONV2D, LibTorchBinding.NN_CONV2D_DESC);
            MemorySegment biasHandle = bias != null ? bias.nativeHandle() : MemorySegment.NULL;
            MemorySegment result = (MemorySegment) fn.invoke(
                    input.nativeHandle(), weight.nativeHandle(), biasHandle,
                    stride, padding, dilation, groups, 0L);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("conv2d failed", t);
        }
    }

    // ── Normalization ─────────────────────────────────────────────────

    /**
     * Batch normalization.
     */
    public static TorchTensor batchNorm(TorchTensor input, TorchTensor weight, TorchTensor bias,
            TorchTensor runningMean, TorchTensor runningVar,
            boolean training, double momentum, double eps) {
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(
                    LibTorchBinding.NN_BATCH_NORM, LibTorchBinding.NN_BATCH_NORM_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(
                    input.nativeHandle(),
                    weight != null ? weight.nativeHandle() : MemorySegment.NULL,
                    bias != null ? bias.nativeHandle() : MemorySegment.NULL,
                    runningMean != null ? runningMean.nativeHandle() : MemorySegment.NULL,
                    runningVar != null ? runningVar.nativeHandle() : MemorySegment.NULL,
                    training, momentum, eps);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("batch_norm failed", t);
        }
    }

    /**
     * Layer normalization.
     *
     * @param input           input tensor
     * @param normalizedShape shape of the last N dims to normalize
     * @param weight          optional learnable weight (gamma)
     * @param bias            optional learnable bias (beta)
     * @param eps             epsilon for numerical stability
     * @return normalized tensor
     */
    public static TorchTensor layerNorm(TorchTensor input, long[] normalizedShape,
            TorchTensor weight, TorchTensor bias, double eps) {
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(
                    LibTorchBinding.NN_LAYER_NORM, LibTorchBinding.NN_LAYER_NORM_DESC);
            MemorySegment shapeSegment = opArena.allocateFrom(ValueLayout.JAVA_LONG, normalizedShape);
            MemorySegment result = (MemorySegment) fn.invoke(
                    input.nativeHandle(),
                    shapeSegment, (long) normalizedShape.length,
                    weight != null ? weight.nativeHandle() : MemorySegment.NULL,
                    bias != null ? bias.nativeHandle() : MemorySegment.NULL,
                    eps);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("layer_norm failed", t);
        }
    }

    // ── Pooling ───────────────────────────────────────────────────────

    /**
     * 2D max pooling.
     *
     * @param input      input tensor [N, C, H, W]
     * @param kernelSize pooling window size
     * @param stride     stride of the pooling window
     * @param padding    implicit zero-padding
     * @return pooled tensor
     */
    public static TorchTensor maxPool2d(TorchTensor input, long kernelSize, long stride, long padding) {
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(
                    "at_max_pool2d",
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
            MemorySegment result = (MemorySegment) fn.invoke(
                    input.nativeHandle(), kernelSize, stride, padding);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("max_pool2d failed", t);
        }
    }

    /**
     * Adaptive average pooling 2D.
     */
    public static TorchTensor adaptiveAvgPool2d(TorchTensor input, long outputH, long outputW) {
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(
                    "at_adaptive_avg_pool2d",
                    FunctionDescriptor.of(ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
            MemorySegment result = (MemorySegment) fn.invoke(
                    input.nativeHandle(), outputH, outputW);
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("adaptive_avg_pool2d failed", t);
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────

    private static TorchTensor invokeUnary(String fnName, TorchTensor input) {
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(
                    fnName, LibTorchBinding.TENSOR_UNARY_OP_DESC);
            MemorySegment result = (MemorySegment) fn.invoke(input.nativeHandle());
            return new TorchTensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException(fnName + " failed", t);
        }
    }
}
