package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * Mish activation function.
 * <p>
 * Mish is a self-regularizing, non-monotonic smooth activation function that has shown
 * excellent performance in modern deep learning models, especially transformers and
 * computer vision architectures. It's differentiable and provides good gradient flow.
 * <p>
 * {@code Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))}
 * <p>
 * Mish is particularly effective in transformer models and YOLOv4 object detection.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * Mish(x) = x * tanh(softplus(x))
 *         = x * tanh(ln(1 + exp(x)))
 * </pre>
 *
 * <h3>Characteristics</h3>
 * <ul>
 *   <li>Smooth and non-monotonic (not always increasing)</li>
 *   <li>Self-regularizing without explicit dropout</li>
 *   <li>Non-bounded: can output values beyond input range</li>
 *   <li>Well-behaved gradients in both positive and negative regions</li>
 *   <li>Slightly more computationally expensive than ReLU</li>
 * </ul>
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Input:</b> arbitrary shape</li>
 *   <li><b>Output:</b> same shape as input</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var mish = new Mish();
 * var output = mish.forward(input);
 * }</pre>
 *
 * <h3>When to Use</h3>
 * <ul>
 *   <li>Transformer architectures (often outperforms GELU/ReLU)</li>
 *   <li>Modern computer vision models</li>
 *   <li>When you want smooth gradient flow without explicit dropout</li>
 *   <li>Object detection networks (YOLOv4-style)</li>
 * </ul>
 *
 * <h3>Comparison to Other Activations</h3>
 * <ul>
 *   <li>vs ReLU: Mish is smoother and self-regularizing</li>
 *   <li>vs GELU: Mish often performs comparably or better</li>
 *   <li>vs SiLU: Mish is more smooth and less prone to outliers</li>
 *   <li>vs ELU: Mish has better gradient properties overall</li>
 * </ul>
 *
 * <h3>Performance Notes</h3>
 * Mish typically requires 2-3 exp() and tanh() calls. For inference-heavy applications,
 * consider using faster activations like ReLU. For training deep models, Mish often
 * converges faster and achieves better final accuracy.
 */
public class Mish extends NNModule {

    /**
     * Create a Mish activation function.
     */
    public Mish() {
    }

    /**
     * Apply Mish activation to input.
     *
     * @param input tensor of arbitrary shape
     * @return output tensor with same shape
     */
    @Override
    public GradTensor forward(GradTensor input) {
        float[] data = input.data();
        float[] result = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            // softplus(x) = ln(1 + exp(x))
            float softplus = (float) Math.log(1 + Math.exp(data[i]));
            // Mish(x) = x * tanh(softplus(x))
            result[i] = data[i] * (float) Math.tanh(softplus);
        }

        GradTensor out = GradTensor.of(result, input.shape());
        if (input.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("Mish") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] ug = upstream.data();
                    float[] grad = new float[data.length];
                    for (int i = 0; i < data.length; i++) {
                        // softplus(x) = ln(1 + exp(x))
                        float exp_x = (float) Math.exp(Math.min(data[i], 20)); // cap to avoid overflow
                        float softplus = (float) Math.log(1 + exp_x);
                        // tanh(u) and sech^2(u) = 1 - tanh^2(u)
                        float tanh_softplus = (float) Math.tanh(softplus);
                        float sech2 = 1 - tanh_softplus * tanh_softplus;
                        // d(softplus)/dx = exp(x) / (1 + exp(x)) = sigmoid(x)
                        float sigmoid_x = exp_x / (1 + exp_x);
                        // d(Mish)/dx = tanh(softplus) + x * sech^2(softplus) * sigmoid
                        grad[i] = ug[i] * (tanh_softplus + data[i] * sech2 * sigmoid_x);
                    }
                    input.backward(GradTensor.of(grad, input.shape()));
                }
            });
        }
        return out;
    }

    @Override
    public String toString() {
        return "Mish()";
    }
}
