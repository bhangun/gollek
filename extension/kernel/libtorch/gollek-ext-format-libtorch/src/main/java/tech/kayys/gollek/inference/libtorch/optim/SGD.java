package tech.kayys.gollek.inference.libtorch.optim;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.core.Tensor;

import java.lang.invoke.MethodHandle;
import java.util.List;
import java.util.Optional;

/**
 * Stochastic Gradient Descent optimizer with momentum and weight decay.
 */
public class SGD extends Optimizer {

    private final double momentum;
    private final double dampening;
    private final double weightDecay;
    private final boolean nesterov;

    public SGD(List<Tensor> parameters, double learningRate,
            double momentum, double dampening,
            double weightDecay, boolean nesterov) {
        super(parameters, learningRate);
        this.momentum = momentum;
        this.dampening = dampening;
        this.weightDecay = weightDecay;
        this.nesterov = nesterov;
    }

    /** Convenience constructor with momentum only. */
    public SGD(List<Tensor> parameters, double learningRate, double momentum) {
        this(parameters, learningRate, momentum, 0.0, 0.0, false);
    }

    /** Convenience constructor with defaults. */
    public SGD(List<Tensor> parameters, double learningRate) {
        this(parameters, learningRate, 0.0, 0.0, 0.0, false);
    }

    @Override
    public void step() {
        LibTorchBinding binding = LibTorchBinding.getInstance();
        Optional<MethodHandle> sgdStep = binding.bindOptional(LibTorchBinding.SGD_STEP, LibTorchBinding.SGD_STEP_DESC);

        for (Tensor param : parameters) {
            try (Tensor grad = param.grad()) {
                if (grad == null)
                    continue;

                if (sgdStep.isPresent()) {
                    // Native optimized path
                    sgdStep.get().invoke(
                            param.nativeHandle(), grad.nativeHandle(),
                            learningRate, momentum, dampening, weightDecay, nesterov);
                } else {
                    // The native SGD step function is not available.
                    // Full Java fallback would require in-place tensor ops
                    // which are not yet exposed. Log and skip.
                    throw new UnsupportedOperationException(
                            "SGD step requires native at_sgd_step symbol. "
                                    + "Ensure the LibTorch wrapper library is built.");
                }
            } catch (UnsupportedOperationException e) {
                throw e;
            } catch (Throwable t) {
                throw new RuntimeException("SGD step failed for parameter", t);
            }
        }
    }
}
