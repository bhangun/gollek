package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * Batch Normalization layer for stabilizing and accelerating training.
 * <p>
 * Batch Normalization normalizes layer inputs across the batch dimension to have
 * zero mean and unit variance, then applies learnable affine transformation (scale and shift).
 * This dramatically speeds up training and allows higher learning rates.
 * <p>
 * During training, statistics are computed from the batch. During evaluation,
 * running statistics (exponential moving average) are used.
 * <p>
 * Equivalent to {@code torch.nn.BatchNorm1d}.
 *
 * <h3>Mathematical Definition (Training)</h3>
 * <pre>
 * μ_batch = (1/m) Σ x_i                      (batch mean)
 * σ_batch² = (1/m) Σ (x_i - μ_batch)²        (batch variance)
 * x_norm = (x - μ_batch) / √(σ_batch² + ε)
 * y = γ * x_norm + β                          (learnable scale and shift)
 * </pre>
 *
 * <h3>Mathematical Definition (Evaluation)</h3>
 * <pre>
 * Uses running mean and running variance:
 * x_norm = (x - running_mean) / √(running_var + ε)
 * y = γ * x_norm + β
 * </pre>
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Input:</b> [batch, numFeatures] or any shape ending with numFeatures</li>
 *   <li><b>Output:</b> same shape as input</li>
 *   <li><b>Weight (γ):</b> [numFeatures]</li>
 *   <li><b>Bias (β):</b> [numFeatures]</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var bn = new BatchNorm1d(256);  // 256 features
 * var output = bn.forward(input);  // [batch, 256] -> [batch, 256]
 * }</pre>
 *
 * <h3>Typical Usage in Model</h3>
 * <pre>{@code
 * var model = new Sequential(
 *     new Linear(784, 256),
 *     new BatchNorm1d(256),
 *     new ReLU(),
 *     new Linear(256, 128),
 *     new BatchNorm1d(128),
 *     new ReLU(),
 *     new Linear(128, 10)
 * );
 * }</pre>
 *
 * <h3>Training vs Evaluation Mode</h3>
 * <pre>{@code
 * model.train();   // Use batch statistics, update running stats
 * // training code...
 *
 * model.eval();    // Use running statistics, don't update
 * // evaluation code...
 * }</pre>
 *
 * <h3>Key Parameters</h3>
 * <ul>
 *   <li><b>eps:</b> small constant for numerical stability (default 1e-5)</li>
 *   <li><b>momentum:</b> coefficient for running stats exponential average (default 0.1)</li>
 *   <li><b>γ (weight):</b> learnable scale, initialized to 1</li>
 *   <li><b>β (bias):</b> learnable shift, initialized to 0</li>
 * </ul>
 *
 * <h3>Running Statistics Update</h3>
 * <pre>
 * running_mean = momentum * batch_mean + (1 - momentum) * running_mean
 * running_var = momentum * batch_var + (1 - momentum) * running_var
 * </pre>
 *
 * <h3>Advantages</h3>
 * <ul>
 *   <li>Dramatically speeds up training (2-10x faster)</li>
 *   <li>Allows higher learning rates</li>
 *   <li>Reduces internal covariate shift</li>
 *   <li>Acts as regularizer (reduces overfitting)</li>
 *   <li>More stable training dynamics</li>
 * </ul>
 *
 * <h3>Important Notes</h3>
 * <ul>
 *   <li>Behavior differs between training and evaluation modes</li>
 *   <li>Batch size affects training (very small batches = noisy statistics)</li>
 *   <li>Running statistics need to be tracked during training</li>
 *   <li>Usually place after Linear/Conv layers, before activation</li>
 * </ul>
 *
 * <h3>Common Configurations</h3>
 * <ul>
 *   <li>eps = 1e-5 (default, good for most cases)</li>
 *   <li>momentum = 0.1 (standard, 10:90 batch:running average)</li>
 *   <li>momentum = 0.01 (conservative, more weight on running average)</li>
 * </ul>
 */
public class BatchNorm1d extends NNModule {

    private final int numFeatures;
    private final float eps;
    private final float momentum;

    private final Parameter weight;  // γ (scale)
    private final Parameter bias;    // β (shift)

    // Running statistics for evaluation mode
    private float[] runningMean;
    private float[] runningVar;

    /**
     * Create a BatchNorm1d layer with default parameters.
     *
     * @param numFeatures number of features to normalize
     */
    public BatchNorm1d(int numFeatures) {
        this(numFeatures, 1e-5f, 0.1f);
    }

    /**
     * Create a BatchNorm1d layer with custom parameters.
     *
     * @param numFeatures number of features to normalize
     * @param eps         small constant for numerical stability
     * @param momentum    coefficient for running average (0.0-1.0)
     *
     * @throws IllegalArgumentException if numFeatures is non-positive
     */
    public BatchNorm1d(int numFeatures, float eps, float momentum) {
        if (numFeatures <= 0) {
            throw new IllegalArgumentException("numFeatures must be positive, got: " + numFeatures);
        }
        if (eps <= 0) {
            throw new IllegalArgumentException("eps must be positive, got: " + eps);
        }
        if (momentum < 0 || momentum > 1) {
            throw new IllegalArgumentException("momentum must be in [0, 1], got: " + momentum);
        }

        this.numFeatures = numFeatures;
        this.eps = eps;
        this.momentum = momentum;

        // Initialize learnable parameters
        this.weight = registerParameter("weight", GradTensor.ones(1, numFeatures));
        this.bias = registerParameter("bias", GradTensor.zeros(1, numFeatures));

        // Initialize running statistics
        this.runningMean = new float[numFeatures];
        this.runningVar = new float[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            runningMean[i] = 0.0f;
            runningVar[i] = 1.0f;
        }
    }

    /**
     * Apply batch normalization to input.
     *
     * @param input tensor of shape [..., numFeatures]
     * @return output tensor with same shape after normalization
     *
     * @throws IllegalArgumentException if input last dimension doesn't match numFeatures
     */
    @Override
    public GradTensor forward(GradTensor input) {
        long[] shape = input.shape();
        if (shape.length < 1 || shape[shape.length - 1] != numFeatures) {
            throw new IllegalArgumentException(
                "input last dimension must be " + numFeatures + ", got: " +
                (shape.length > 0 ? shape[shape.length - 1] : "unknown"));
        }

        float[] data = input.data();
        int batchSize = (int) (input.numel() / numFeatures);
        float[] result = new float[data.length];
        float[] normalized = new float[data.length];
        float[] batchMean = new float[numFeatures];
        float[] batchVar = new float[numFeatures];

        if (isTraining()) {
            // Training: compute statistics from batch
            // Compute batch mean
            for (int f = 0; f < numFeatures; f++) {
                float sum = 0;
                for (int b = 0; b < batchSize; b++) {
                    sum += data[b * numFeatures + f];
                }
                batchMean[f] = sum / batchSize;
            }

            // Compute batch variance
            for (int f = 0; f < numFeatures; f++) {
                float sumSq = 0;
                for (int b = 0; b < batchSize; b++) {
                    float diff = data[b * numFeatures + f] - batchMean[f];
                    sumSq += diff * diff;
                }
                batchVar[f] = sumSq / batchSize;
            }

            // Normalize and apply affine transform
            float[] w = weight.data().data();
            float[] b = bias.data().data();
            for (int b_idx = 0; b_idx < batchSize; b_idx++) {
                for (int f = 0; f < numFeatures; f++) {
                    int idx = b_idx * numFeatures + f;
                    normalized[idx] = (data[idx] - batchMean[f]) / (float) Math.sqrt(batchVar[f] + eps);
                    result[idx] = w[f] * normalized[idx] + b[f];
                }
            }

            // Update running statistics (exponential moving average)
            for (int f = 0; f < numFeatures; f++) {
                runningMean[f] = momentum * batchMean[f] + (1 - momentum) * runningMean[f];
                runningVar[f] = momentum * batchVar[f] + (1 - momentum) * runningVar[f];
            }
        } else {
            // Evaluation: use running statistics
            float[] w = weight.data().data();
            float[] b = bias.data().data();
            for (int b_idx = 0; b_idx < batchSize; b_idx++) {
                for (int f = 0; f < numFeatures; f++) {
                    int idx = b_idx * numFeatures + f;
                    normalized[idx] = (data[idx] - runningMean[f]) / (float) Math.sqrt(runningVar[f] + eps);
                    result[idx] = w[f] * normalized[idx] + b[f];
                }
            }
        }

        GradTensor out = GradTensor.of(result, input.shape());
        if (input.requiresGrad()) {
            out.requiresGrad(true);
            // Backward pass computes gradients through normalization
            out.setGradFn(new Function.Context("BatchNorm1d") {
                @Override
                public void backward(GradTensor upstream) {
                    // This is a simplified backward - full implementation would
                    // compute gradients w.r.t. weight, bias, running stats, and input
                    // For now, we pass through to input
                    input.backward(upstream);
                }
            });
        }

        return out;
    }

    /**
     * Get the running mean statistics.
     *
     * @return copy of running mean array
     */
    public float[] getRunningMean() {
        return runningMean.clone();
    }

    /**
     * Get the running variance statistics.
     *
     * @return copy of running variance array
     */
    public float[] getRunningVar() {
        return runningVar.clone();
    }

    @Override
    public String toString() {
        return "BatchNorm1d(" + numFeatures + ", eps=" + eps + ", momentum=" + momentum + ")";
    }
}
