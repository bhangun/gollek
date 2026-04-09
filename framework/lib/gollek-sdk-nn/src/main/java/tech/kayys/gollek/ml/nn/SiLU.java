package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.Function;

/**
 * Sigmoid Linear Unit (SiLU), also known as Swish activation function.
 * <p>
 * Applies the self-gating activation: {@code y = x * σ(x)} where σ is the sigmoid function.
 * <p>
 * SiLU provides smooth non-linearity with good gradient flow. It has become popular
 * in modern architectures like EfficientNet, MobileNetV3, and recent language models
 * due to its improved performance over ReLU in many tasks.
 * <p>
 * Equivalent to {@code torch.nn.SiLU}.
 *
 * <h3>Mathematical Definition</h3>
 * <pre>
 * SiLU(x) = x * σ(x) = x / (1 + exp(-x))
 * </pre>
 *
 * <h3>Gradient</h3>
 * <pre>
 * dSiLU/dx = σ(x) + x * σ(x) * (1 - σ(x))
 *          = σ(x) * (1 + x * (1 - σ(x)))
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
 * var silu = new SiLU();
 * var output = silu.forward(input);
 * }</pre>
 *
 * <h3>Typical Usage</h3>
 * <pre>{@code
 * var model = new Sequential(
 *     new Linear(512, 1024),
 *     new SiLU(),
 *     new Linear(1024, 256)
 * );
 * }</pre>
 *
 * <h3>Characteristics</h3>
 * <ul>
 *   <li>Self-gating: the activation of each neuron is modulated by its input</li>
 *   <li>Smooth and non-monotonic (unlike ReLU)</li>
 *   <li>Better gradient propagation than ReLU</li>
 *   <li>Computationally slightly more expensive than ReLU</li>
 *   <li>Often provides better accuracy at the cost of slight overhead</li>
 * </ul>
 *
 * <h3>Historical Note</h3>
 * Originally named "Swish" in "Searching for Activation Functions" paper (2017),
 * later standardized as "SiLU" (Sigmoid Linear Unit) in PyTorch.
 */
public class SiLU extends NNModule {

    /**
     * Create a SiLU activation function.
     */
    public SiLU() {
    }

    /**
     * Apply SiLU activation to input.
     *
     * @param input tensor of arbitrary shape
     * @return output tensor with same shape after SiLU transformation
     */
    @Override
    public GradTensor forward(GradTensor input) {
        float[] data = input.data();
        float[] result = new float[data.length];
        float[] sigData = new float[data.length];

        // Forward pass: compute sigmoid and SiLU for each element
        for (int i = 0; i < data.length; i++) {
            sigData[i] = 1.0f / (1.0f + (float) Math.exp(-data[i]));
            result[i] = data[i] * sigData[i];
        }

        GradTensor out = GradTensor.of(result, input.shape());
        if (input.requiresGrad()) {
            out.requiresGrad(true);
            out.setGradFn(new Function.Context("SiLU") {
                @Override
                public void backward(GradTensor upstream) {
                    float[] ug = upstream.data();
                    float[] grad = new float[data.length];
                    for (int i = 0; i < data.length; i++) {
                        // d/dx [x * σ(x)] = σ(x) + x * σ(x) * (1 - σ(x))
                        grad[i] = ug[i] * (sigData[i] + data[i] * sigData[i] * (1 - sigData[i]));
                    }
                    input.backward(GradTensor.of(grad, input.shape()));
                }
            });
        }
        return out;
    }

    @Override
    public String toString() {
        return "SiLU()";
    }
}
