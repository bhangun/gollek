package io.github.pytorch.nn.functional;

import io.github.pytorch.core.Tensor;
import io.github.pytorch.ffm.LibTorchFFM;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Neural network functional operations (torch.nn.functional equivalent)
 */
public class Functional {
    
    /**
     * ReLU activation function
     */
    public static Tensor relu(Tensor input) {
        return relu(input, false);
    }
    
    public static Tensor relu(Tensor input, boolean inplace) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment result = (MemorySegment) LibTorchFFM.NN_RELU.invoke(
                input.handle(), inplace
            );
            return new Tensor(result, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("ReLU failed", e);
        }
    }
    
    /**
     * GELU activation function
     */
    public static Tensor gelu(Tensor input) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment result = (MemorySegment) LibTorchFFM.NN_GELU.invoke(
                input.handle()
            );
            return new Tensor(result, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("GELU failed", e);
        }
    }
    
    /**
     * Dropout
     */
    public static Tensor dropout(Tensor input, double p, boolean training) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment result = (MemorySegment) LibTorchFFM.NN_DROPOUT.invoke(
                input.handle(), p, training
            );
            return new Tensor(result, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Dropout failed", e);
        }
    }
    
    /**
     * Softmax
     */
    public static Tensor softmax(Tensor input, int dim) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment result = (MemorySegment) LibTorchFFM.NN_SOFTMAX.invoke(
                input.handle(), dim
            );
            return new Tensor(result, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Softmax failed", e);
        }
    }
    
    /**
     * Log softmax
     */
    public static Tensor logSoftmax(Tensor input, int dim) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment result = (MemorySegment) LibTorchFFM.NN_LOG_SOFTMAX.invoke(
                input.handle(), dim
            );
            return new Tensor(result, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Log softmax failed", e);
        }
    }
    
    /**
     * Cross entropy loss
     */
    public static Tensor crossEntropy(Tensor input, Tensor target) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment result = (MemorySegment) LibTorchFFM.NN_CROSS_ENTROPY.invoke(
                input.handle(), target.handle(), MemorySegment.NULL, 1 // mean reduction
            );
            return new Tensor(result, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Cross entropy failed", e);
        }
    }
    
    /**
     * MSE loss
     */
    public static Tensor mseLoss(Tensor input, Tensor target) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment result = (MemorySegment) LibTorchFFM.NN_MSE_LOSS.invoke(
                input.handle(), target.handle(), 1 // mean reduction
            );
            return new Tensor(result, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("MSE loss failed", e);
        }
    }
    
    /**
     * Linear transformation
     */
    public static Tensor linear(Tensor input, Tensor weight, Tensor bias) {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment result = (MemorySegment) LibTorchFFM.NN_LINEAR.invoke(
                input.handle(),
                weight.handle(),
                bias != null ? bias.handle() : MemorySegment.NULL
            );
            return new Tensor(result, arena);
        } catch (Throwable e) {
            arena.close();
            throw new RuntimeException("Linear failed", e);
        }
    }
}
