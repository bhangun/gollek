package io.quarkus.pytorch.optim;

import io.quarkus.pytorch.core.Tensor;

import java.util.List;
import java.util.ArrayList;

/**
 * Base class for all optimizers.
 * Optimizers update model parameters based on computed gradients.
 */
public abstract class Optimizer {
    
    protected final List<Tensor> parameters;
    protected final double learningRate;
    
    /**
     * Create an optimizer.
     * 
     * @param parameters List of parameters to optimize
     * @param learningRate Learning rate
     */
    public Optimizer(List<Tensor> parameters, double learningRate) {
        this.parameters = new ArrayList<>(parameters);
        this.learningRate = learningRate;
    }
    
    /**
     * Perform a single optimization step.
     */
    public abstract void step();
    
    /**
     * Zero all parameter gradients.
     */
    public void zeroGrad() {
        for (Tensor param : parameters) {
            Tensor grad = param.grad();
            if (grad != null) {
                // Reset gradient to zero
                grad.close();
            }
        }
    }
    
    /**
     * Get learning rate.
     */
    public double getLearningRate() {
        return learningRate;
    }
    
    /**
     * Get parameters.
     */
    public List<Tensor> getParameters() {
        return new ArrayList<>(parameters);
    }
}
