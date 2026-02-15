package io.quarkus.pytorch.nn;

import io.quarkus.pytorch.core.Tensor;
import io.quarkus.pytorch.core.Tensor.ScalarType;

/**
 * Fully connected (Linear) layer: y = xW^T + b
 * Applies a linear transformation to incoming data.
 */
public class Linear extends Module {
    
    private final long inFeatures;
    private final long outFeatures;
    private final boolean bias;
    
    private Tensor weight;
    private Tensor biasParam;
    
    /**
     * Create a linear layer.
     * 
     * @param inFeatures Size of input features
     * @param outFeatures Size of output features
     * @param bias If true, adds a learnable bias
     */
    public Linear(long inFeatures, long outFeatures, boolean bias) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.bias = bias;
        
        reset();
    }
    
    /**
     * Create a linear layer with bias.
     */
    public Linear(long inFeatures, long outFeatures) {
        this(inFeatures, outFeatures, true);
    }
    
    /**
     * Initialize weights using Kaiming uniform initialization.
     */
    private void reset() {
        // Initialize weights: uniform(-sqrt(k), sqrt(k)) where k = 1/in_features
        double k = 1.0 / inFeatures;
        double bound = Math.sqrt(k);
        
        weight = Tensor.rand(new long[]{outFeatures, inFeatures}, ScalarType.FLOAT);
        // Scale to [-bound, bound]
        weight = weight.mul(Tensor.zeros(new long[]{1}, ScalarType.FLOAT).add(2 * bound))
                      .add(-bound);
        weight.requiresGrad(true);
        registerParameter("weight", weight);
        
        if (bias) {
            biasParam = Tensor.rand(new long[]{outFeatures}, ScalarType.FLOAT);
            biasParam = biasParam.mul(Tensor.zeros(new long[]{1}, ScalarType.FLOAT).add(2 * bound))
                                .add(-bound);
            biasParam.requiresGrad(true);
            registerParameter("bias", biasParam);
        }
    }
    
    @Override
    public Tensor forward(Tensor input) {
        // input: (*, in_features)
        // weight: (out_features, in_features)
        // output: (*, out_features)
        
        Tensor output = input.matmul(weight.transpose(0, 1));
        
        if (bias) {
            output = output.add(biasParam);
        }
        
        return output;
    }
    
    public long getInFeatures() {
        return inFeatures;
    }
    
    public long getOutFeatures() {
        return outFeatures;
    }
    
    public boolean hasBias() {
        return bias;
    }
    
    public Tensor getWeight() {
        return weight;
    }
    
    public Tensor getBias() {
        return biasParam;
    }
    
    @Override
    public String toString() {
        return String.format("Linear(in_features=%d, out_features=%d, bias=%s)",
            inFeatures, outFeatures, bias);
    }
}
