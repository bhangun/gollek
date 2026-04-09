package tech.kayys.gollek.sdk.vision.layers;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;

/**
 * ReLU (Rectified Linear Unit) activation.
 *
 * <p>Applies the rectified linear unit function element-wise: ReLU(x) = max(0, x)</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * ReLU relu = new ReLU();
 * GradTensor x = GradTensor.of(new float[]{-1, 2, -3, 4}, 2, 2);
 * GradTensor y = relu.forward(x);  // [0, 2, 0, 4]
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class ReLU extends NNModule {

    /**
     * Forward pass.
     *
     * @param input input tensor
     * @return output tensor with ReLU applied element-wise
     */
    public GradTensor forward(GradTensor input) {
        float[] inputData = input.data();
        float[] outputData = new float[inputData.length];

        for (int i = 0; i < inputData.length; i++) {
            outputData[i] = Math.max(0.0f, inputData[i]);
        }

        return GradTensor.of(outputData, input.shape());
    }

    /**
     * Apply ReLU to a tensor (static utility method).
     *
     * @param x input tensor
     * @return output tensor with ReLU applied
     */
    public static GradTensor apply(GradTensor x) {
        float[] inputData = x.data();
        float[] outputData = new float[inputData.length];

        for (int i = 0; i < inputData.length; i++) {
            outputData[i] = Math.max(0.0f, inputData[i]);
        }

        return GradTensor.of(outputData, x.shape());
    }
}
