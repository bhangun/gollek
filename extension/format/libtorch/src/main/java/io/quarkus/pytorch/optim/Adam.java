package io.quarkus.pytorch.optim;

import io.quarkus.pytorch.core.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Adam (Adaptive Moment Estimation) optimizer.
 * Computes adaptive learning rates for each parameter using estimates
 * of first and second moments of the gradients.
 */
public class Adam extends Optimizer {
    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double eps;
    private final double weightDecay;
    private final boolean amsgrad;
    
    private final Map<Tensor, Tensor> expAvg = new HashMap<>();  // First moment
    private final Map<Tensor, Tensor> expAvgSq = new HashMap<>();  // Second moment
    private final Map<Tensor, Tensor> maxExpAvgSq = new HashMap<>();  // For AMSGrad
    private int step = 0;
    
    /**
     * Create Adam optimizer.
     * 
     * @param parameters Parameters to optimize
     * @param learningRate Learning rate
     * @param beta1 Coefficient for computing running average of gradient
     * @param beta2 Coefficient for computing running average of squared gradient
     * @param eps Term added to denominator for numerical stability
     * @param weightDecay Weight decay (L2 penalty)
     * @param amsgrad Use AMSGrad variant
     */
    public Adam(List<Tensor> parameters, double learningRate,
                double beta1, double beta2, double eps,
                double weightDecay, boolean amsgrad) {
        super(parameters, learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.amsgrad = amsgrad;
    }


    public Adam(List<Tensor> parameters, double learningRate, 
                double beta1, double beta2, double eps, double weightDecay, boolean amsgrad) {
        super(parameters);
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.amsgrad = amsgrad;
        
        try {
            // Create parameter array
            MemorySegment paramsArray = arena.allocate(
                ValueLayout.ADDRESS, parameters.size()
            );
            for (int i = 0; i < parameters.size(); i++) {
                paramsArray.setAtIndex(ValueLayout.ADDRESS, i, parameters.get(i).handle());
            }
            
            // Create native Adam optimizer
            nativeHandle = (MemorySegment) LibTorchFFM.OPTIM_ADAM_NEW.invoke(
                paramsArray, parameters.size(), learningRate, beta1, beta2, 
                eps, weightDecay, amsgrad
            );
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create Adam optimizer", e);
        }
    }
    
    
    /**
     * Create Adam optimizer with default parameters.
     */
    public Adam(List<Tensor> parameters, double learningRate) {
        this(parameters, learningRate, 0.9, 0.999, 1e-8, 0, false);
    }
    
    @Override
    public void step() {
        step++;
        
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
            
            // Initialize moment estimates
            if (!expAvg.containsKey(param)) {
                expAvg.put(param, Tensor.zeros(param.shape(), Tensor.ScalarType.FLOAT));
                expAvgSq.put(param, Tensor.zeros(param.shape(), Tensor.ScalarType.FLOAT));
                if (amsgrad) {
                    maxExpAvgSq.put(param, Tensor.zeros(param.shape(), Tensor.ScalarType.FLOAT));
                }
            }
            
            Tensor m = expAvg.get(param);
            Tensor v = expAvgSq.get(param);
            
            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
            m = m.mul(Tensor.ones(new long[]{1}, Tensor.ScalarType.FLOAT).add(beta1))
                 .add(grad.mul(Tensor.ones(new long[]{1}, Tensor.ScalarType.FLOAT).add(1 - beta1)));
            
            // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
            v = v.mul(Tensor.ones(new long[]{1}, Tensor.ScalarType.FLOAT).add(beta2))
                 .add(grad.mul(grad).mul(Tensor.ones(new long[]{1}, Tensor.ScalarType.FLOAT).add(1 - beta2)));
            
            expAvg.put(param, m);
            expAvgSq.put(param, v);
            
            // Compute bias-corrected moment estimates
            double biasCorrection1 = 1.0 - Math.pow(beta1, step);
            double biasCorrection2 = 1.0 - Math.pow(beta2, step);
            
            Tensor mHat = m.mul(Tensor.ones(new long[]{1}, Tensor.ScalarType.FLOAT)
                .add(1.0 / biasCorrection1));
            Tensor vHat = v.mul(Tensor.ones(new long[]{1}, Tensor.ScalarType.FLOAT)
                .add(1.0 / biasCorrection2));
            
            if (amsgrad) {
                Tensor maxV = maxExpAvgSq.get(param);
                // maxV = max(maxV, vHat)
                // This requires element-wise max operation
                maxExpAvgSq.put(param, vHat);  // Simplified
                vHat = maxExpAvgSq.get(param);
            }
            
            // Update parameters: param = param - lr * mHat / (sqrt(vHat) + eps)
            // This requires square root and division operations
            
            throw new UnsupportedOperationException(
                "Adam step requires additional tensor operations (sqrt, div, in-place update)");
        }
    }
    
    @Override
    public String toString() {
        return String.format("Adam(lr=%g, beta1=%g, beta2=%g, eps=%g, weight_decay=%g, amsgrad=%s)",
            learningRate, beta1, beta2, eps, weightDecay, amsgrad);
    }
}
