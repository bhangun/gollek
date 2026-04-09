package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * Randomly zeros elements with probability {@code p} during training.
 * <p>
 * Dropout is a regularization technique that helps prevent overfitting by
 * randomly disabling neurons during training. During evaluation mode, no dropout
 * is applied. The output is scaled by 1/(1-p) during training (inverted dropout)
 * so that the expected value remains unchanged.
 * <p>
 * Equivalent to {@code torch.nn.Dropout}.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * Training:
 *   mask[i] ~ Bernoulli(1-p)
 *   output[i] = input[i] * mask[i] / (1-p)
 *
 * Evaluation:
 *   output[i] = input[i]  (no dropout)
 * </pre>
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Input:</b> arbitrary shape</li>
 *   <li><b>Output:</b> same shape as input</li>
 * </ul>
 *
 * <h3>Example: Training</h3>
 * <pre>{@code
 * var dropout = new Dropout(0.5);  // drop 50% of neurons
 * model.train();
 * var output = dropout.forward(input);  // random masking applied
 * }</pre>
 *
 * <h3>Example: Evaluation</h3>
 * <pre>{@code
 * model.eval();
 * var output = dropout.forward(input);  // no masking, output == input
 * }</pre>
 *
 * <h3>Typical Usage in Models</h3>
 * <pre>{@code
 * var model = new Sequential(
 *     new Linear(784, 256),
 *     new ReLU(),
 *     new Dropout(0.2),
 *     new Linear(256, 128),
 *     new ReLU(),
 *     new Dropout(0.2),
 *     new Linear(128, 10)
 * );
 * }</pre>
 *
 * <h3>Key Characteristics</h3>
 * <ul>
 *   <li>Only active during training (isTraining() == true)</li>
 *   <li>Uses inverted dropout for consistent training/eval behavior</li>
 *   <li>Stochastic: different mask generated for each forward pass</li>
 *   <li>Common dropout rates: 0.1-0.5 depending on layer depth</li>
 * </ul>
 */
public class Dropout extends NNModule {

    private final float p;
    private final java.util.Random rng = new java.util.Random();

    /**
     * Create a dropout layer with specified drop probability.
     *
     * @param p probability of dropping each element (must be in [0, 1])
     *
     * @throws IllegalArgumentException if p is not in [0, 1]
     */
    public Dropout(float p) {
        if (p < 0 || p > 1) {
            throw new IllegalArgumentException("dropout probability must be in [0, 1], got: " + p);
        }
        this.p = p;
    }

    /**
     * Apply dropout to input during training, pass through during evaluation.
     *
     * @param input tensor of arbitrary shape
     * @return output tensor (same shape, possibly with some elements zeroed)
     */
    @Override
    public GradTensor forward(GradTensor input) {
        // During eval mode or p=0, no dropout applied
        if (!isTraining() || p == 0) {
            return input;
        }

        float[] data = input.data();
        float[] result = new float[data.length];
        boolean[] mask = new boolean[data.length];
        float scale = 1.0f / (1.0f - p);  // Inverted dropout scaling

        // Generate random mask and apply it
        for (int i = 0; i < data.length; i++) {
            mask[i] = rng.nextFloat() >= p;
            result[i] = mask[i] ? data[i] * scale : 0;
        }

        GradTensor out = GradTensor.of(result, input.shape());
        if (input.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("Dropout") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] ug = upstream.data();
                    float[] grad = new float[data.length];
                    // Apply same mask to gradients (zeros for dropped positions)
                    for (int i = 0; i < data.length; i++) {
                        grad[i] = mask[i] ? ug[i] * scale : 0;
                    }
                    input.backward(GradTensor.of(grad, input.shape()));
                }
            });
        }
        return out;
    }

    /**
     * Get the dropout probability.
     *
     * @return p value
     */
    public float getP() {
        return p;
    }

    @Override
    public String toString() {
        return "Dropout(p=" + p + ")";
    }
}
