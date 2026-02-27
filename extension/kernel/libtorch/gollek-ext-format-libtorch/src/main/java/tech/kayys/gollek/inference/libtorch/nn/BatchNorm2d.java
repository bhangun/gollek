package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.core.Device;
import tech.kayys.gollek.inference.libtorch.core.ScalarType;
import tech.kayys.gollek.inference.libtorch.core.Tensor;

/**
 * 2D Batch Normalization layer.
 * Mirrors {@code torch::nn::BatchNorm2d}.
 */
public class BatchNorm2d extends Module {

    private final long numFeatures;
    private final double eps;
    private final double momentum;
    private final boolean affine;
    private final boolean trackRunningStats;

    /**
     * Create a BatchNorm2d layer.
     *
     * @param numFeatures       number of feature channels (C in [N, C, H, W])
     * @param eps               epsilon for numerical stability
     * @param momentum          value used for running mean/var computation
     * @param affine            if true, learnable affine parameters (gamma, beta)
     * @param trackRunningStats if true, tracks running mean and variance
     */
    public BatchNorm2d(long numFeatures, double eps, double momentum,
            boolean affine, boolean trackRunningStats) {
        this.numFeatures = numFeatures;
        this.eps = eps;
        this.momentum = momentum;
        this.affine = affine;
        this.trackRunningStats = trackRunningStats;

        if (affine) {
            // gamma (weight) initialized to 1, beta (bias) initialized to 0
            registerParameter("weight", Tensor.ones(new long[] { numFeatures }, ScalarType.FLOAT, Device.CPU));
            registerParameter("bias", Tensor.zeros(new long[] { numFeatures }, ScalarType.FLOAT, Device.CPU));
        }

        if (trackRunningStats) {
            registerBuffer("running_mean", Tensor.zeros(new long[] { numFeatures }, ScalarType.FLOAT, Device.CPU));
            registerBuffer("running_var", Tensor.ones(new long[] { numFeatures }, ScalarType.FLOAT, Device.CPU));
        }
    }

    /** Convenience constructor with standard defaults. */
    public BatchNorm2d(long numFeatures) {
        this(numFeatures, 1e-5, 0.1, true, true);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor weight = affine ? parameters.get("weight") : null;
        Tensor bias = affine ? parameters.get("bias") : null;
        Tensor runningMean = trackRunningStats ? buffers.get("running_mean") : null;
        Tensor runningVar = trackRunningStats ? buffers.get("running_var") : null;

        return Functional.batchNorm(input, weight, bias, runningMean, runningVar,
                training, momentum, eps);
    }

    @Override
    public String toString() {
        return String.format("BatchNorm2d(%d, eps=%.1e, momentum=%.2f, affine=%b)",
                numFeatures, eps, momentum, affine);
    }
}
