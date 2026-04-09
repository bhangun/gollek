package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * Cosine Embedding Loss for similarity learning.
 * <p>
 * Encourages embeddings of similar pairs to be close and dissimilar pairs to be far apart
 * using cosine distance as the metric.
 * <p>
 * For positive pairs (y=1): {@code loss = 1 - cos(x1, x2)}
 * For negative pairs (y=-1): {@code loss = max(0, cos(x1, x2) - margin)}
 * <p>
 * Equivalent to {@code torch.nn.CosineEmbeddingLoss}.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var loss = new CosineEmbeddingLoss(0.5f);
 * var x1 = GradTensor.randn(32, 128);
 * var x2 = GradTensor.randn(32, 128);
 * var targets = GradTensor.of(new float[]{1, -1, 1, ...}, 32);
 * var lossTensor = loss.compute(x1, x2, targets);
 * }</pre>
 */
public class CosineEmbeddingLoss {

    private final float margin;

    /**
     * Create a CosineEmbeddingLoss with default margin of 0.0.
     */
    public CosineEmbeddingLoss() {
        this(0.0f);
    }

    /**
     * Create a CosineEmbeddingLoss with specified margin.
     *
     * @param margin margin for dissimilar pairs (default: 0.0). For negative pairs,
     *               loss is max(0, cos(x1, x2) - margin)
     */
    public CosineEmbeddingLoss(float margin) {
        this.margin = margin;
    }

    /**
     * Compute cosine embedding loss.
     *
     * @param x1     first embedding tensor [batch, dim] with requiresGrad=true
     * @param x2     second embedding tensor [batch, dim] with requiresGrad=true
     * @param target labels tensor [batch]: 1 for similar pairs, -1 for dissimilar pairs
     * @return scalar loss tensor with gradient support for backpropagation
     *
     * @throws IllegalArgumentException if x1 and x2 shapes do not match
     * @throws IllegalArgumentException if target batch size does not match x1 batch size
     */
    public GradTensor compute(GradTensor x1, GradTensor x2, GradTensor target) {
        // Validate inputs
        long[] s1 = x1.shape(), s2 = x2.shape();
        if (s1.length != 2 || s2.length != 2) {
            throw new IllegalArgumentException(
                "Expected x1 and x2 to be 2D tensors [batch, dim], got shapes: " +
                java.util.Arrays.toString(s1) + " and " + java.util.Arrays.toString(s2));
        }
        if (s1[0] != s2[0] || s1[1] != s2[1]) {
            throw new IllegalArgumentException(
                "x1 and x2 must have same shape, got: " + java.util.Arrays.toString(s1) +
                " vs " + java.util.Arrays.toString(s2));
        }
        if (target.shape()[0] != s1[0]) {
            throw new IllegalArgumentException(
                "target batch size must match x1 batch size, got: " + target.shape()[0] +
                " vs " + s1[0]);
        }

        float[] d1 = x1.data(), d2 = x2.data(), t = target.data();
        int batch = (int) s1[0];
        int dim = (int) s1[1];

        float[] losses = new float[batch];
        float totalLoss = 0;

        // Forward pass: compute loss for each sample in batch
        for (int b = 0; b < batch; b++) {
            int off = b * dim;
            float dot = 0, norm1 = 0, norm2 = 0;

            // Compute dot product and norms
            for (int i = 0; i < dim; i++) {
                float v1 = d1[off + i];
                float v2 = d2[off + i];
                dot += v1 * v2;
                norm1 += v1 * v1;
                norm2 += v2 * v2;
            }

            // Compute cosine similarity: cos(x1, x2) = dot / (||x1|| * ||x2||)
            float norm1Sqrt = (float) Math.sqrt(norm1);
            float norm2Sqrt = (float) Math.sqrt(norm2);
            float cosine = dot / (norm1Sqrt * norm2Sqrt + 1e-8f);

            // Compute loss: 1 - cos for positive pairs, max(0, cos - margin) for negative
            if (t[b] > 0) {
                losses[b] = 1 - cosine;
            } else {
                losses[b] = Math.max(0, cosine - margin);
            }
            totalLoss += losses[b];
        }

        float avgLoss = totalLoss / batch;

        // Create output tensor with automatic differentiation support
        GradTensor out = GradTensor.scalar(avgLoss);

        // Register backward function for gradient computation
        if (x1.requiresGrad() || x2.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("CosineEmbeddingLoss") {
                @Override
                public void backward(GradTensor upstream) {
                    float scale = upstream.item() / batch;
                    float[] grad1 = new float[d1.length];
                    float[] grad2 = new float[d2.length];

                    // Compute gradients for each sample
                    for (int b = 0; b < batch; b++) {
                        int off = b * dim;

                        // Recompute norms and cosine for this sample
                        float dot = 0, norm1 = 0, norm2 = 0;
                        for (int i = 0; i < dim; i++) {
                            float v1 = d1[off + i];
                            float v2 = d2[off + i];
                            dot += v1 * v2;
                            norm1 += v1 * v1;
                            norm2 += v2 * v2;
                        }

                        float norm1Sqrt = (float) Math.sqrt(norm1);
                        float norm2Sqrt = (float) Math.sqrt(norm2);
                        float eps = 1e-8f;

                        // Compute gradient only if loss is non-zero
                        boolean isPositive = t[b] > 0;
                        boolean isActive = isPositive || (losses[b] > 0);

                        if (isActive) {
                            // Derivative of cosine w.r.t x1 and x2
                            float cosine = dot / (norm1Sqrt * norm2Sqrt + eps);

                            for (int i = 0; i < dim; i++) {
                                float v1 = d1[off + i];
                                float v2 = d2[off + i];

                                // Gradient for x1
                                float dCosine_dx1 = (v2 / (norm2Sqrt + eps) - cosine * v1 / (norm1Sqrt + eps)) /
                                                   (norm1Sqrt + eps);
                                float dLoss_dx1 = isPositive ? -dCosine_dx1 : dCosine_dx1;
                                grad1[off + i] = scale * dLoss_dx1;

                                // Gradient for x2
                                float dCosine_dx2 = (v1 / (norm1Sqrt + eps) - cosine * v2 / (norm2Sqrt + eps)) /
                                                   (norm2Sqrt + eps);
                                float dLoss_dx2 = isPositive ? -dCosine_dx2 : dCosine_dx2;
                                grad2[off + i] = scale * dLoss_dx2;
                            }
                        }
                    }

                    if (x1.requiresGrad()) {
                        x1.backward(GradTensor.of(grad1, x1.shape()));
                    }
                    if (x2.requiresGrad()) {
                        x2.backward(GradTensor.of(grad2, x2.shape()));
                    }
                }
            });
        }

        return out;
    }

    /**
     * Get the margin value.
     *
     * @return the margin for dissimilar pairs
     */
    public float getMargin() {
        return margin;
    }

    @Override
    public String toString() {
        return "CosineEmbeddingLoss(margin=" + margin + ")";
    }
}
