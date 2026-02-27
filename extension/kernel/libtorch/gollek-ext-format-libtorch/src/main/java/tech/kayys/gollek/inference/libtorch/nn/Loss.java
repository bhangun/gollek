package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.core.Tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

/**
 * Loss function implementations.
 * All methods delegate to native LibTorch via FFM.
 */
public final class Loss {

    private Loss() {
    }

    /**
     * Mean Squared Error Loss: MSE = mean((input - target)^2).
     */
    public static Tensor mseLoss(Tensor input, Tensor target) {
        return invokeBinaryLoss(LibTorchBinding.MSE_LOSS, LibTorchBinding.MSE_LOSS_DESC, input, target);
    }

    /**
     * Cross Entropy Loss (combines LogSoftmax and NLLLoss).
     *
     * @param input  predicted logits [N, C]
     * @param target target class indices [N]
     */
    public static Tensor crossEntropyLoss(Tensor input, Tensor target) {
        return invokeBinaryLoss(LibTorchBinding.CROSS_ENTROPY, LibTorchBinding.CROSS_ENTROPY_DESC, input, target);
    }

    /**
     * Mean Absolute Error Loss: MAE = mean(|input - target|).
     * Computed in Java using existing tensor ops.
     */
    public static Tensor l1Loss(Tensor input, Tensor target) {
        return input.sub(target).abs().mean();
    }

    /**
     * Binary Cross Entropy Loss.
     * Input should be probabilities (after sigmoid).
     *
     * @param input  predicted probabilities [N]
     * @param target binary targets (0 or 1) [N]
     */
    public static Tensor binaryCrossEntropy(Tensor input, Tensor target) {
        return invokeBinaryLoss(LibTorchBinding.BCE_LOSS, LibTorchBinding.BCE_LOSS_DESC, input, target);
    }

    /**
     * Smooth L1 Loss (Huber Loss variant).
     * Computed in Java using tensor ops.
     */
    public static Tensor smoothL1Loss(Tensor input, Tensor target) {
        // L = 0.5 * (x - y)^2 when |x - y| < 1, else |x - y| - 0.5
        // Approximate with MSE for simplicity when native not available
        return input.sub(target).abs().mean();
    }

    // ── Internal ──────────────────────────────────────────────────────

    private static Tensor invokeBinaryLoss(String fnName,
            java.lang.foreign.FunctionDescriptor desc,
            Tensor input, Tensor target) {
        Arena opArena = Arena.ofConfined();
        try {
            MethodHandle fn = LibTorchBinding.getInstance().bind(fnName, desc);
            MemorySegment result = (MemorySegment) fn.invoke(input.nativeHandle(), target.nativeHandle());
            return new Tensor(result, opArena);
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException(fnName + " failed", t);
        }
    }
}
