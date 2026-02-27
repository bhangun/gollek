package tech.kayys.gollek.inference.libtorch.optim;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.core.Tensor;

import java.lang.invoke.MethodHandle;
import java.util.List;
import java.util.Optional;

/**
 * Adam optimizer (Adaptive Moment Estimation).
 * Supports weight decay and AMSGrad variant.
 */
public class Adam extends Optimizer {

    private final double beta1;
    private final double beta2;
    private final double eps;
    private final double weightDecay;
    private final boolean amsgrad;
    private long stepCount = 0;

    public Adam(List<Tensor> parameters, double learningRate,
            double beta1, double beta2, double eps,
            double weightDecay, boolean amsgrad) {
        super(parameters, learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.amsgrad = amsgrad;
    }

    /** Convenience constructor with standard defaults. */
    public Adam(List<Tensor> parameters, double learningRate) {
        this(parameters, learningRate, 0.9, 0.999, 1e-8, 0.0, false);
    }

    @Override
    public void step() {
        stepCount++;
        LibTorchBinding binding = LibTorchBinding.getInstance();
        Optional<MethodHandle> adamStep = binding.bindOptional(LibTorchBinding.ADAM_STEP,
                LibTorchBinding.ADAM_STEP_DESC);

        if (adamStep.isEmpty()) {
            throw new UnsupportedOperationException(
                    "Adam step requires native at_adam_step symbol. "
                            + "Ensure the LibTorch wrapper library is built.");
        }

        // Note: exp_avg and exp_avg_sq state tensors are managed natively
        // The native function handles first/second moment tracking internally
        for (Tensor param : parameters) {
            try (Tensor grad = param.grad()) {
                if (grad == null)
                    continue;

                adamStep.get().invoke(
                        param.nativeHandle(), grad.nativeHandle(),
                        // exp_avg and exp_avg_sq are managed by native side
                        param.nativeHandle(), param.nativeHandle(),
                        stepCount,
                        learningRate, beta1, beta2, eps,
                        weightDecay, amsgrad);
            } catch (Throwable t) {
                throw new RuntimeException("Adam step failed", t);
            }
        }
    }
}
