package io.github.pytorch.optim;

import io.github.pytorch.core.Tensor;
import io.github.pytorch.ffm.LibTorchFFM;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * Stochastic Gradient Descent optimizer
 */
public class SGD extends Optimizer {
    
    private final double learningRate;
    private final double momentum;
    private final double weightDecay;
    private final double dampening;
    private final boolean nesterov;
    
    public SGD(List<Tensor> parameters, double learningRate) {
        this(parameters, learningRate, 0.0, 0.0, 0.0, false);
    }
    
    public SGD(List<Tensor> parameters, double learningRate, double momentum) {
        this(parameters, learningRate, momentum, 0.0, 0.0, false);
    }
    
    public SGD(List<Tensor> parameters, double learningRate, double momentum, 
               double weightDecay, double dampening, boolean nesterov) {
        super(parameters);
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.weightDecay = weightDecay;
        this.dampening = dampening;
        this.nesterov = nesterov;
        
        try {
            // Create parameter array
            MemorySegment paramsArray = arena.allocate(
                ValueLayout.ADDRESS, parameters.size()
            );
            for (int i = 0; i < parameters.size(); i++) {
                paramsArray.setAtIndex(ValueLayout.ADDRESS, i, parameters.get(i).handle());
            }
            
            // Create native SGD optimizer
            nativeHandle = (MemorySegment) LibTorchFFM.OPTIM_SGD_NEW.invoke(
                paramsArray, parameters.size(), learningRate, momentum, 
                weightDecay, dampening, nesterov
            );
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create SGD optimizer", e);
        }
    }
    
    @Override
    public void step() {
        checkClosed();
        try {
            LibTorchFFM.OPTIM_SGD_STEP.invoke(nativeHandle);
        } catch (Throwable e) {
            throw new RuntimeException("SGD step failed", e);
        }
    }
    
    @Override
    public void zeroGrad() {
        checkClosed();
        try {
            LibTorchFFM.OPTIM_SGD_ZERO_GRAD.invoke(nativeHandle);
        } catch (Throwable e) {
            throw new RuntimeException("SGD zero_grad failed", e);
        }
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public double getMomentum() {
        return momentum;
    }
    
    @Override
    public String toString() {
        return String.format("SGD(lr=%.4f, momentum=%.2f, weight_decay=%.4f, nesterov=%b)",
            learningRate, momentum, weightDecay, nesterov);
    }
}
