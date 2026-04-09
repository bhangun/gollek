package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Applies a linear transformation: {@code y = x @ W^T + b}.
 * <p>
 * A fully-connected layer that applies a linear transformation to its input.
 * Weight matrix is initialized using Kaiming uniform distribution for better
 * training convergence.
 * <p>
 * Equivalent to {@code torch.nn.Linear} in PyTorch.
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Input:</b> {@code [..., inFeatures]} (arbitrary batch dimensions supported)</li>
 *   <li><b>Weight:</b> {@code [outFeatures, inFeatures]}</li>
 *   <li><b>Bias:</b> {@code [1, outFeatures]} (optional)</li>
 *   <li><b>Output:</b> {@code [..., outFeatures]}</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var linear = new Linear(768, 256);
 * var output = linear.forward(input);  // [batch, 768] → [batch, 256]
 * }</pre>
 *
 * <h3>Parameters</h3>
 * <ul>
 *   <li><b>weight</b> - [outFeatures, inFeatures] learnable weight matrix</li>
 *   <li><b>bias</b> - [1, outFeatures] learnable bias (optional)</li>
 * </ul>
 */
public class Linear extends NNModule {

    private final int inFeatures;
    private final int outFeatures;
    private final boolean useBias;
    private final Parameter weight;
    private final Parameter bias;

    /**
     * Create a linear layer with bias.
     *
     * @param inFeatures  size of input features
     * @param outFeatures size of output features
     *
     * @throws IllegalArgumentException if inFeatures or outFeatures is non-positive
     */
    public Linear(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, true);
    }

    /**
     * Create a linear layer with optional bias.
     *
     * @param inFeatures  size of input features (must be positive)
     * @param outFeatures size of output features (must be positive)
     * @param bias        if true, learn an additive bias; if false, no bias
     *
     * @throws IllegalArgumentException if inFeatures or outFeatures is non-positive
     */
    public Linear(int inFeatures, int outFeatures, boolean bias) {
        if (inFeatures <= 0) {
            throw new IllegalArgumentException("inFeatures must be positive, got: " + inFeatures);
        }
        if (outFeatures <= 0) {
            throw new IllegalArgumentException("outFeatures must be positive, got: " + outFeatures);
        }

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.useBias = bias;

        // Kaiming uniform initialization: U(-bound, bound) where bound = 1/sqrt(inFeatures)
        float bound = (float) (1.0 / Math.sqrt(inFeatures));
        float[] wData = new float[outFeatures * inFeatures];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < wData.length; i++) {
            wData[i] = (rng.nextFloat() * 2 - 1) * bound;
        }
        this.weight = registerParameter("weight", GradTensor.of(wData, outFeatures, inFeatures));

        if (bias) {
            float[] bData = new float[outFeatures];
            for (int i = 0; i < bData.length; i++) {
                bData[i] = (rng.nextFloat() * 2 - 1) * bound;
            }
            this.bias = registerParameter("bias", GradTensor.of(bData, 1, outFeatures));
        } else {
            this.bias = null;
        }
    }

    /**
     * Apply the linear transformation.
     *
     * @param input tensor of shape {@code [..., inFeatures]}
     * @return output tensor of shape {@code [..., outFeatures]}
     *
     * @throws IllegalArgumentException if input last dimension does not match inFeatures
     */
    @Override
    public GradTensor forward(GradTensor input) {
        long[] shape = input.shape();
        if (shape.length < 1) {
            throw new IllegalArgumentException("input must be at least 1D");
        }
        if (shape[shape.length - 1] != inFeatures) {
            throw new IllegalArgumentException(
                "input last dimension must be " + inFeatures + ", got: " + shape[shape.length - 1]);
        }

        // input: [..., inFeatures], weight: [outFeatures, inFeatures]
        // result = input @ weight^T + bias → [..., outFeatures]
        GradTensor result = input.matmul(weight.data().transpose());
        if (useBias && bias != null) {
            result = result.add(bias.data());
        }
        return result;
    }

    /**
     * Get the number of input features.
     *
     * @return inFeatures
     */
    public int getInFeatures() {
        return inFeatures;
    }

    /**
     * Get the number of output features.
     *
     * @return outFeatures
     */
    public int getOutFeatures() {
        return outFeatures;
    }

    /**
     * Check if this layer has a bias term.
     *
     * @return true if bias is enabled
     */
    public boolean hasBias() {
        return useBias;
    }

    @Override
    public String toString() {
        return "Linear(in=" + inFeatures + ", out=" + outFeatures + ", bias=" + useBias + ")";
    }
}
