package io.quarkus.pytorch.optim;

import io.quarkus.pytorch.core.Tensor;

import java.util.List;
import java.util.HashMap;
import java.util.Map;

/**
 * Stochastic Gradient Descent optimizer with optional momentum.
 * Updates parameters using: param = param - lr * grad
 * With momentum: velocity = momentum * velocity + grad
 *                param = param - lr * velocity
 */
public class SGD extends Optimizer {
    
    private final double momentum;
    private final double dampening;
    private final double weightDecay;
    private final boolean nesterov;
    
    private final Map<Tensor, Tensor> velocityBuffer = new HashMap<>();
    
    /**
     * Create SGD optimizer.
     * 
     * @param parameters Parameters to optimize
     * @param learningRate Learning rate
     * @param momentum Momentum factor (0 to disable)
     * @param dampening Dampening for momentum
     * @param weightDecay Weight decay (L2 penalty)
     * @param nesterov Enable Nesterov momentum
     */
    public SGD(List<Tensor> parameters, double learningRate, double momentum,
               double dampening, double weightDecay, boolean nesterov) {
        super(parameters, learningRate);
        this.momentum = momentum;
        this.dampening = dampening;
        this.weightDecay = weightDecay;
        this.nesterov = nesterov;
        
        if (nesterov && (momentum <= 0 || dampening != 0)) {
            throw new IllegalArgumentException(
                "Nesterov momentum requires a momentum and zero dampening");
        }
    }
    
    /**
     * Create SGD optimizer with default parameters.
     */
    public SGD(List<Tensor> parameters, double learningRate) {
        this(parameters, learningRate, 0, 0, 0, false);
    }
    
    /**
     * Create SGD optimizer with momentum.
     */
    public SGD(List<Tensor> parameters, double learningRate, double momentum) {
        this(parameters, learningRate, momentum, 0, 0, false);
    }
    
    @Override
    public void step() {
        for (Tensor param : parameters) {
            Tensor grad = param.grad();
            if (grad == null) {
                continue;
            }
            
            // Apply weight decay
            if (weightDecay != 0) {
                grad = grad.add(param.mul(Tensor.ones(new long[]{1}, 
                    Tensor.ScalarType.FLOAT).add(weightDecay)));
            }
            
            if (momentum != 0) {
                Tensor velocity;
                if (!velocityBuffer.containsKey(param)) {
                    // Initialize velocity
                    velocity = grad;
                    velocityBuffer.put(param, velocity);
                } else {
                    velocity = velocityBuffer.get(param);
                    // velocity = momentum * velocity + (1 - dampening) * grad
                    velocity = velocity.mul(Tensor.ones(new long[]{1}, 
                        Tensor.ScalarType.FLOAT).add(momentum))
                        .add(grad.mul(Tensor.ones(new long[]{1}, 
                            Tensor.ScalarType.FLOAT).add(1 - dampening)));
                    velocityBuffer.put(param, velocity);
                }
                
                if (nesterov) {
                    grad = grad.add(velocity.mul(Tensor.ones(new long[]{1}, 
                        Tensor.ScalarType.FLOAT).add(momentum)));
                } else {
                    grad = velocity;
                }
            }
            
            // Update parameter: param = param - lr * grad
            Tensor update = grad.mul(Tensor.ones(new long[]{1}, 
                Tensor.ScalarType.FLOAT).add(-learningRate));
            
            // This would require in-place addition
            // param.addInplace(update);
            
            throw new UnsupportedOperationException(
                "SGD step requires in-place parameter update operations");
        }
    }
    
    @Override
    public String toString() {
        return String.format("SGD(lr=%g, momentum=%g, dampening=%g, weight_decay=%g, nesterov=%s)",
            learningRate, momentum, dampening, weightDecay, nesterov);
    }
}
