package io.github.pytorch.nn;

import io.github.pytorch.core.Tensor;
import io.github.pytorch.ffm.LibTorchFFM;
import java.lang.foreign.MemorySegment;

/**
 * Linear (fully connected) layer
 * Applies: y = xW^T + b
 */
public class Linear extends Module {
    
    private final int inFeatures;
    private final int outFeatures;
    private final boolean hasBias;
    
    public Linear(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, true);
    }
    
    public Linear(int inFeatures, int outFeatures, boolean bias) {
        super();
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.hasBias = bias;
        
        // Initialize weight: (out_features, in_features)
        Tensor weight = Tensor.randn(new long[]{outFeatures, inFeatures});
        registerParameter("weight", weight);
        
        // Initialize bias if needed
        if (bias) {
            Tensor biasParam = Tensor.zeros(new long[]{outFeatures});
            registerParameter("bias", biasParam);
        }
    }
    
    @Override
    public Tensor forward(Tensor input) {
        Tensor weight = getParameter("weight");
        Tensor bias = hasBias ? getParameter("bias") : null;
        
        try {
            MemorySegment result = (MemorySegment) LibTorchFFM.NN_LINEAR.invoke(
                input.handle(),
                weight.handle(),
                bias != null ? bias.handle() : MemorySegment.NULL
            );
            
            return new Tensor(result, arena);
        } catch (Throwable e) {
            throw new RuntimeException("Linear forward failed", e);
        }
    }
    
    public int inFeatures() {
        return inFeatures;
    }
    
    public int outFeatures() {
        return outFeatures;
    }
    
    @Override
    public String toString() {
        return String.format("Linear(in_features=%d, out_features=%d, bias=%b)",
            inFeatures, outFeatures, hasBias);
    }
}
