package tech.kayys.gollek.sdk.optimize;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.List;

/**
 * Base class for all optimizers.
 *
 * <p>Optimizers implement the standard optimization algorithms (SGD, Adam, AdamW, etc.)
 * to update model parameters based on computed gradients.</p>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * Optimizer optimizer = new Adam(model.parameters(), 0.001);
 *
 * // In training loop:
 * loss.backward();
 * optimizer.step();
 * optimizer.zeroGrad();
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public abstract class Optimizer {

    protected final List<GradTensor> parameters;
    protected final double lr;
    protected int step = 0;

    /**
     * Create optimizer.
     *
     * @param parameters list of parameters to optimize
     * @param lr         learning rate
     */
    protected Optimizer(List<GradTensor> parameters, double lr) {
        this.parameters = parameters;
        this.lr = lr;
    }

    /**
     * Perform a single optimization step.
     */
    public abstract void step();

    /**
     * Clear gradients of all optimized parameters.
     */
    public void zeroGrad() {
        for (GradTensor param : parameters) {
            param.zeroGrad();
        }
    }

    /**
     * Get current learning rate.
     *
     * @return learning rate
     */
    public double getLr() {
        return lr;
    }

    /**
     * Set learning rate.
     *
     * @param lr new learning rate
     */
    public abstract void setLr(double lr);

    /**
     * Get current step number.
     *
     * @return step count
     */
    public int getStep() {
        return step;
    }

    /**
     * Get list of optimized parameters.
     *
     * @return parameter list
     */
    public List<GradTensor> getParameters() {
        return parameters;
    }

    /**
     * Clip gradients by global norm.
     *
     * @param maxNorm maximum allowed gradient norm
     */
    public void clipGradNorm(double maxNorm) {
        double totalNorm = 0.0;
        for (GradTensor param : parameters) {
            if (param.grad() != null) {
                float[] grad = param.grad().data();
                for (float g : grad) {
                    totalNorm += g * g;
                }
            }
        }
        totalNorm = Math.sqrt(totalNorm);

        if (totalNorm > maxNorm) {
            double scale = maxNorm / (totalNorm + 1e-6);
            for (GradTensor param : parameters) {
                if (param.grad() != null) {
                    float[] grad = param.grad().data();
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] *= (float) scale;
                    }
                }
            }
        }
    }

    /**
     * Clip gradients by value.
     *
     * @param minVal minimum gradient value
     * @param maxVal maximum gradient value
     */
    public void clipGradValue(float minVal, float maxVal) {
        for (GradTensor param : parameters) {
            if (param.grad() != null) {
                float[] grad = param.grad().data();
                for (int i = 0; i < grad.length; i++) {
                    grad[i] = Math.max(minVal, Math.min(maxVal, grad[i]));
                }
            }
        }
    }
}
