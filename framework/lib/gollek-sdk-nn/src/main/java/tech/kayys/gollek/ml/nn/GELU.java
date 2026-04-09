package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * Gaussian Error Linear Unit (GELU) activation function.
 * <p>
 * {@code GELU(x) = x * Φ(x)} where Φ(x) is the cumulative distribution function
 * of the standard normal distribution. GELU is a smooth activation function that
 * has become popular in transformer-based models like BERT and GPT.
 * <p>
 * This implementation uses the tanh approximation for numerical stability:
 * {@code 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))}
 * <p>
 * The approximation is numerically stable and very close to the exact CDF-based computation.
 * <p>
 * Equivalent to {@code torch.nn.GELU}.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * </pre>
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Input:</b> arbitrary shape</li>
 *   <li><b>Output:</b> same shape as input</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var gelu = new GELU();
 * var output = gelu.forward(input);
 * }</pre>
 *
 * <h3>Typical Usage in Transformers</h3>
 * <pre>{@code
 * var feedforward = new Sequential(
 *     new Linear(768, 3072),  // expand
 *     new GELU(),             // non-linearity
 *     new Linear(3072, 768)   // contract
 * );
 * }</pre>
 *
 * <h3>Characteristics</h3>
 * <ul>
 *   <li>Smooth activation (differentiable everywhere)</li>
 *   <li>Self-gating property: output depends on input magnitude</li>
 *   <li>Good gradient flow for deep networks</li>
 *   <li>Slightly more computationally expensive than ReLU</li>
 * </ul>
 */
public class GELU extends NNModule {

    /**
     * Create a GELU activation function.
     */
    public GELU() {
    }

    /**
     * Apply GELU activation to input.
     *
     * @param input tensor of arbitrary shape
     * @return output tensor with same shape after GELU transformation
     */
    @Override
    public GradTensor forward(GradTensor input) {
        float[] data = input.data();
        float[] result = new float[data.length];
        float sqrt2OverPi = (float) Math.sqrt(2.0 / Math.PI);

        // Forward pass: compute GELU(x) for each element
        for (int i = 0; i < data.length; i++) {
            float x = data[i];
            float inner = sqrt2OverPi * (x + 0.044715f * x * x * x);
            result[i] = 0.5f * x * (1.0f + (float) Math.tanh(inner));
        }

        GradTensor out = GradTensor.of(result, input.shape());
        if (input.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("GELU") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] ug = upstream.data();
                    float[] grad = new float[data.length];
                    for (int i = 0; i < data.length; i++) {
                        float x = data[i];
                        float inner = sqrt2OverPi * (x + 0.044715f * x * x * x);
                        float tanhVal = (float) Math.tanh(inner);
                        // d(tanh(u))/du = 1 - tanh²(u) = sech²(u)
                        float sech2 = 1.0f - tanhVal * tanhVal;
                        // du/dx = √(2/π) * (1 + 3 * 0.044715 * x²)
                        float dInner = sqrt2OverPi * (1.0f + 3.0f * 0.044715f * x * x);
                        // d(GELU)/dx = 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * d.../dx
                        grad[i] = ug[i] * (0.5f * (1.0f + tanhVal) + 0.5f * x * sech2 * dInner);
                    }
                    input.backward(GradTensor.of(grad, input.shape()));
                }
            });
        }
        return out;
    }

    @Override
    public String toString() {
        return "GELU()";
    }
}
