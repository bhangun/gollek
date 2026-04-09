package tech.kayys.gollek.sdk.vision.layers;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendRegistry;
import tech.kayys.gollek.sdk.vision.backend.VisionBackendProvider;

/**
 * Linear (fully connected) layer.
 *
 * <p>Applies a linear transformation to the incoming data: y = xA^T + b</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Linear fc = new Linear(512, 1000);  // 512 input features, 1000 output classes
 * GradTensor x = GradTensor.randn(1, 512);
 * GradTensor y = fc.forward(x);  // Shape: [1, 1000]
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class Linear extends NNModule {

    private final int inFeatures;
    private final int outFeatures;
    private final boolean bias;

    // Learnable parameters
    private final Parameter weight;
    private final Parameter biasParam;

    /**
     * Create a linear layer.
     *
     * @param inFeatures  number of input features
     * @param outFeatures number of output features
     * @param bias        if true, adds a learnable bias to the output
     */
    public Linear(int inFeatures, int outFeatures, boolean bias) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.bias = bias;

        // Initialize weights with Kaiming uniform initialization
        this.weight = registerParameter("weight", kaimingUniform(outFeatures, inFeatures));
        
        if (bias) {
            float[] biasData = new float[outFeatures];
            // Initialize bias to zeros
            this.biasParam = registerParameter("bias", GradTensor.of(biasData, outFeatures));
        } else {
            this.biasParam = null;
        }
    }

    /**
     * Create Linear with bias enabled.
     */
    public Linear(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, true);
    }

    @Override
    public GradTensor forward(GradTensor input) {
        // Get the appropriate backend for execution
        VisionBackendProvider backend = VisionBackendRegistry.getDefault();
        
        // Delegate to backend
        return backend.linear(input, weight.data(), bias ? biasParam.data() : null);
    }

    /**
     * Get the weight tensor.
     *
     * @return weight tensor of shape [out_features, in_features]
     */
    public GradTensor getWeight() {
        return weight.data();
    }

    /**
     * Get the bias tensor.
     *
     * @return bias tensor of shape [out_features], or null if bias is disabled
     */
    public GradTensor getBias() {
        return biasParam != null ? biasParam.data() : null;
    }

    /**
     * Get input features.
     */
    public int getInFeatures() {
        return inFeatures;
    }

    /**
     * Get output features.
     */
    public int getOutFeatures() {
        return outFeatures;
    }

    /**
     * Kaiming uniform initialization.
     *
     * @param outCh output features
     * @param inCh  input features
     * @return initialized tensor
     */
    private static GradTensor kaimingUniform(int outCh, int inCh) {
        int fanIn = inCh;
        double std = Math.sqrt(2.0 / fanIn);
        double bound = std * Math.sqrt(3.0);

        int size = outCh * inCh;
        float[] data = new float[size];
        java.util.Random rng = new java.util.Random(42);

        for (int i = 0; i < size; i++) {
            data[i] = (float) (rng.nextDouble() * 2 * bound - bound);
        }

        return GradTensor.of(data, outCh, inCh);
    }

    @Override
    public String toString() {
        return String.format("Linear(%d, %d, bias=%b)", inFeatures, outFeatures, bias);
    }
}
