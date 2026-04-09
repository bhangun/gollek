package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

import java.util.Arrays;

/**
 * Layer Normalization (LayerNorm).
 * <p>
 * Normalizes input features to have zero mean and unit variance across the last dimension,
 * then applies a learnable affine transformation. Layer normalization is particularly
 * effective for transformer-based models and provides stability during training.
 * <p>
 * Transformation: {@code y = (x - μ) / √(σ² + ε) * γ + β}
 * where μ and σ are computed per sample across features, γ (weight) and β (bias) are learned.
 * <p>
 * Equivalent to {@code torch.nn.LayerNorm}.
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Input:</b> [*, normalizedShape] (arbitrary batch dimensions)</li>
 *   <li><b>Output:</b> same shape as input</li>
 *   <li><b>Weight (γ):</b> [1, normalizedShape]</li>
 *   <li><b>Bias (β):</b> [1, normalizedShape]</li>
 * </ul>
 *
 * <h3>Examples</h3>
 * <pre>{@code
 * // For transformer: input [batch, seq_len, 768]
 * var layerNorm = new LayerNorm(768);
 * var output = layerNorm.forward(input);  // [batch, seq_len, 768]
 *
 * // For dense layers: input [batch, 256]
 * var ln = new LayerNorm(256);
 * }</pre>
 *
 * <h3>Typical Usage in Transformers</h3>
 * <pre>{@code
 * // Pre-normalization (used in modern transformers)
 * var output = input;
 * output = layerNorm.forward(output);        // normalize first
 * output = multiheadAttn.forward(output);    // then attend
 * output = output.add(input);                // residual connection
 *
 * // Post-normalization (earlier architecture)
 * var output = multiheadAttn.forward(input);
 * output = layerNorm.forward(output.add(input));
 * }</pre>
 *
 * <h3>Key Parameters</h3>
 * <ul>
 *   <li><b>eps</b> - small epsilon (default 1e-5) for numerical stability to avoid division by zero</li>
 *   <li><b>gamma (weight)</b> - learnable scale parameter, initialized to ones</li>
 *   <li><b>beta (bias)</b> - learnable shift parameter, initialized to zeros</li>
 * </ul>
 *
 * <h3>Advantages Over Batch Norm</h3>
 * <ul>
 *   <li>Normalizes per-sample independently (doesn't depend on batch size)</li>
 *   <li>Works well with variable-length sequences</li>
 *   <li>Effective for NLP and sequential models</li>
 *   <li>Behavior identical in training and evaluation modes</li>
 * </ul>
 */
public class LayerNorm extends NNModule {

    private final int normalizedShape;
    private final float eps;
    private final Parameter gamma;  // Weight/scale parameter
    private final Parameter beta;   // Bias/shift parameter

    /**
     * Create a LayerNorm layer with default epsilon (1e-5).
     *
     * @param normalizedShape number of features to normalize over (last dimension)
     *
     * @throws IllegalArgumentException if normalizedShape is non-positive
     */
    public LayerNorm(int normalizedShape) {
        this(normalizedShape, 1e-5f);
    }

    /**
     * Create a LayerNorm layer with specified epsilon.
     *
     * @param normalizedShape number of features to normalize over (last dimension)
     * @param eps             small constant for numerical stability (default: 1e-5)
     *
     * @throws IllegalArgumentException if normalizedShape is non-positive or eps is non-positive
     */
    public LayerNorm(int normalizedShape, float eps) {
        if (normalizedShape <= 0) {
            throw new IllegalArgumentException("normalizedShape must be positive, got: " + normalizedShape);
        }
        if (eps <= 0) {
            throw new IllegalArgumentException("eps must be positive, got: " + eps);
        }

        this.normalizedShape = normalizedShape;
        this.eps = eps;
        // Initialize γ to ones (identity scaling) and β to zeros (no shift)
        this.gamma = registerParameter("weight", GradTensor.ones(1, normalizedShape));
        this.beta = registerParameter("bias", GradTensor.zeros(1, normalizedShape));
    }

    /**
     * Apply layer normalization to input.
     *
     * @param input tensor of shape [*, normalizedShape] where * can be any batch dimensions
     * @return normalized tensor with same shape as input
     *
     * @throws IllegalArgumentException if input last dimension doesn't match normalizedShape
     */
    @Override
    public GradTensor forward(GradTensor input) {
        float[] data = input.data();
        int d = normalizedShape;
        int batches = (int) (input.numel() / d);

        if (batches * d != input.numel()) {
            throw new IllegalArgumentException(
                "input numel (" + input.numel() + ") not divisible by normalizedShape (" + d + ")");
        }

        float[] result = new float[data.length];
        float[] means = new float[batches];
        float[] invStds = new float[batches];
        float[] normalized = new float[data.length];

        // Compute mean and variance per batch element
        for (int b = 0; b < batches; b++) {
            int off = b * d;

            // Compute mean: μ = (1/d) Σ x_i
            float mean = 0;
            for (int i = 0; i < d; i++) {
                mean += data[off + i];
            }
            mean /= d;
            means[b] = mean;

            // Compute variance: σ² = (1/d) Σ (x_i - μ)²
            float var = 0;
            for (int i = 0; i < d; i++) {
                float diff = data[off + i] - mean;
                var += diff * diff;
            }
            var /= d;
            float invStd = 1.0f / (float) Math.sqrt(var + eps);
            invStds[b] = invStd;

            // Normalize and apply affine transform
            float[] gd = gamma.data().data();
            float[] bd = beta.data().data();
            for (int i = 0; i < d; i++) {
                // normalize[i] = (x[i] - μ) / √(σ² + ε)
                normalized[off + i] = (data[off + i] - mean) * invStd;
                // output[i] = γ * normalize[i] + β
                result[off + i] = normalized[off + i] * gd[i % gd.length] + bd[i % bd.length];
            }
        }

        GradTensor out = GradTensor.of(result, input.shape());
        if (input.requiresGrad() || gamma.data().requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("LayerNorm") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] ug = upstream.data();
                    float[] gd = gamma.data().data();

                    // Gradient w.r.t. input
                    if (input.requiresGrad()) {
                        float[] grad = new float[data.length];
                        for (int b = 0; b < batches; b++) {
                            int off = b * d;
                            // Compute two sums needed for input gradients
                            float sumUgGamma = 0, sumUgGammaNorm = 0;
                            for (int i = 0; i < d; i++) {
                                float ugGamma = ug[off + i] * gd[i % gd.length];
                                sumUgGamma += ugGamma;
                                sumUgGammaNorm += ugGamma * normalized[off + i];
                            }
                            // Chain rule through normalization
                            for (int i = 0; i < d; i++) {
                                grad[off + i] = invStds[b] / d * (
                                    d * ug[off + i] * gd[i % gd.length]
                                    - sumUgGamma
                                    - normalized[off + i] * sumUgGammaNorm
                                );
                            }
                        }
                        input.backward(GradTensor.of(grad, input.shape()));
                    }

                    // Gradient w.r.t. gamma (weight)
                    if (gamma.data().requiresGrad()) {
                        float[] gGrad = new float[gd.length];
                        for (int b = 0; b < batches; b++) {
                            int off = b * d;
                            for (int i = 0; i < d; i++) {
                                gGrad[i % gd.length] += ug[off + i] * normalized[off + i];
                            }
                        }
                        gamma.data().backward(GradTensor.of(gGrad, gamma.data().shape()));
                    }

                    // Gradient w.r.t. beta (bias)
                    if (beta.data().requiresGrad()) {
                        float[] bGrad = new float[beta.data().data().length];
                        for (int b = 0; b < batches; b++) {
                            int off = b * d;
                            for (int i = 0; i < d; i++) {
                                bGrad[i % bGrad.length] += ug[off + i];
                            }
                        }
                        beta.data().backward(GradTensor.of(bGrad, beta.data().shape()));
                    }
                }
            });
        }
        return out;
    }

    /**
     * Get the shape being normalized.
     *
     * @return normalizedShape
     */
    public int getNormalizedShape() {
        return normalizedShape;
    }

    /**
     * Get the epsilon value used for numerical stability.
     *
     * @return eps
     */
    public float getEps() {
        return eps;
    }

    @Override
    public String toString() {
        return "LayerNorm(" + normalizedShape + ", eps=" + eps + ")";
    }
}
