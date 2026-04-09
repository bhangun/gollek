package tech.kayys.gollek.sdk.vision.layers;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendRegistry;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendProvider;

import java.util.Arrays;

/**
 * 2D Batch Normalization layer.
 *
 * <p>Applies Batch Normalization over a 4D input (mini-batch of 2D inputs with additional channel dimension).</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * BatchNorm2d bn = new BatchNorm2d(64);  // 64 channels
 * GradTensor x = GradTensor.randn(1, 64, 224, 224);
 * GradTensor y = bn.forward(x);  // Shape: [1, 64, 224, 224]
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class BatchNorm2d extends NNModule {

    private final int numFeatures;
    private final double eps;
    private final double momentum;
    private final boolean affine;
    private final boolean trackRunningStats;

    // Learnable parameters
    private final Parameter weight;
    private final Parameter bias;

    // Running statistics (for inference)
    private float[] runningMean;
    private float[] runningVar;
    private int numBatchesTracked = 0;

    /**
     * Create a 2D batch normalization layer.
     *
     * @param numFeatures         number of features (channels)
     * @param eps                 value added to denominator for numerical stability
     * @param momentum            momentum for running mean and variance
     * @param affine              if true, this module has learnable affine parameters
     * @param trackRunningStats   if true, track running mean and variance
     */
    public BatchNorm2d(int numFeatures, double eps, double momentum, boolean affine, boolean trackRunningStats) {
        this.numFeatures = numFeatures;
        this.eps = eps;
        this.momentum = momentum;
        this.affine = affine;
        this.trackRunningStats = trackRunningStats;

        // Initialize learnable parameters
        if (affine) {
            this.weight = registerParameter("weight", GradTensor.ones(numFeatures));
            this.bias = registerParameter("bias", GradTensor.zeros(numFeatures));
        } else {
            this.weight = null;
            this.bias = null;
        }

        // Initialize running statistics
        if (trackRunningStats) {
            this.runningMean = new float[numFeatures];
            this.runningVar = new float[numFeatures];
            Arrays.fill(this.runningVar, 1.0f);
        }
    }

    /**
     * Create BatchNorm2d with default settings.
     */
    public BatchNorm2d(int numFeatures) {
        this(numFeatures, 1e-5, 0.1, true, true);
    }

    @Override
    public GradTensor forward(GradTensor input) {
        // Get the appropriate backend for execution
        VisionBackendProvider backend = VisionBackendRegistry.getDefault();
        
        // Convert running mean and variance to GradTensors for backend call
        GradTensor runningMeanTensor = trackRunningStats ? GradTensor.of(runningMean, numFeatures) : null;
        GradTensor runningVarTensor = trackRunningStats ? GradTensor.of(runningVar, numFeatures) : null;
        
        // Delegate to backend
        return backend.batchNorm(input, affine ? weight.data() : null, affine ? bias.data() : null,
                                runningMeanTensor, runningVarTensor,
                                isTraining(), (float) momentum, (float) eps);
    }

    /**
     * Set training mode.
     *
     * @param mode true for training, false for inference
     */
    public void setTraining(boolean mode) {
        if (mode) train(); else eval();
    }


    /**
     * Get the weight (gamma) parameter.
     */
    public GradTensor getWeight() {
        return weight != null ? weight.data() : null;
    }

    /**
     * Get the bias (beta) parameter.
     */
    public GradTensor getBias() {
        return bias != null ? bias.data() : null;
    }

    /**
     * Get running mean.
     */
    public float[] getRunningMean() {
        return runningMean;
    }

    /**
     * Get running variance.
     */
    public float[] getRunningVar() {
        return runningVar;
    }

    /**
     * Get number of features.
     */
    public int getNumFeatures() {
        return numFeatures;
    }

    @Override
    public String toString() {
        return String.format("BatchNorm2d(%d, eps=%.1e, momentum=%.2f, affine=%b)",
                numFeatures, eps, momentum, affine);
    }
}
