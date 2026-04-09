package tech.kayys.gollek.sdk.optimize;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.List;
import java.util.HashMap;
import java.util.Map;

/**
 * Adam (Adaptive Moment Estimation) optimizer.
 *
 * <p>Implements the Adam algorithm from "Adam: A Method for Stochastic Optimization"
 * (Kingma & Ba, 2014). Combines the advantages of AdaGrad and RMSProp.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Optimizer optimizer = new Adam(model.parameters(), 0.001)
 *     .betas(0.9, 0.999)
 *     .eps(1e-8)
 *     .weightDecay(0.01)
 *     .build();
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class Adam extends Optimizer {

    private final double beta1;
    private final double beta2;
    private final double eps;
    private final double weightDecay;
    private final boolean amsgrad;

    // Per-parameter state
    private final Map<GradTensor, float[]> expAvg = new HashMap<>();
    private final Map<GradTensor, float[]> expAvgSq = new HashMap<>();
    private final Map<GradTensor, float[]> maxExpAvgSq = new HashMap<>();

    private Adam(Builder builder) {
        super(builder.parameters, builder.lr);
        this.beta1 = builder.beta1;
        this.beta2 = builder.beta2;
        this.eps = builder.eps;
        this.weightDecay = builder.weightDecay;
        this.amsgrad = builder.amsgrad;
    }

    /**
     * Create a new Adam builder.
     *
     * @param parameters list of parameters to optimize
     * @param lr         learning rate
     * @return builder instance
     */
    public static Builder builder(List<GradTensor> parameters, double lr) {
        return new Builder(parameters, lr);
    }

    /**
     * Create Adam with default settings.
     *
     * @param parameters list of parameters to optimize
     * @param lr         learning rate
     * @return optimizer instance
     */
    public static Adam create(List<GradTensor> parameters, double lr) {
        return builder(parameters, lr).build();
    }

    @Override
    public void step() {
        step++;

        double biasCorrection1 = 1.0 - Math.pow(beta1, step);
        double biasCorrection2 = 1.0 - Math.pow(beta2, step);

        for (GradTensor param : parameters) {
            if (param.grad() == null) continue;

            float[] grad = param.grad().data();
            float[] data = param.data();

            // Initialize state if needed
            if (!expAvg.containsKey(param)) {
                expAvg.put(param, new float[data.length]);
                expAvgSq.put(param, new float[data.length]);
                if (amsgrad) {
                    maxExpAvgSq.put(param, new float[data.length]);
                }
            }

            float[] m = expAvg.get(param);
            float[] v = expAvgSq.get(param);

            // Weight decay (AdamW style)
            if (weightDecay != 0) {
                for (int i = 0; i < data.length; i++) {
                    data[i] *= (float) (1.0 - lr * weightDecay);
                }
            }

            // Update biased first and second moment estimates
            for (int i = 0; i < data.length; i++) {
                m[i] = (float) (beta1 * m[i] + (1.0 - beta1) * grad[i]);
                v[i] = (float) (beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i]);
            }

            // Bias-corrected estimates
            float mHatCoeff = (float) (1.0 / biasCorrection1);
            float vHatCoeff = (float) (1.0 / biasCorrection2);

            // AMSGrad variant
            if (amsgrad) {
                float[] vHat = maxExpAvgSq.get(param);
                for (int i = 0; i < data.length; i++) {
                    vHat[i] = Math.max(vHat[i], v[i]);
                    data[i] -= (float) (lr * m[i] * mHatCoeff / (Math.sqrt(vHat[i] * vHatCoeff) + eps));
                }
            } else {
                for (int i = 0; i < data.length; i++) {
                    data[i] -= (float) (lr * m[i] * mHatCoeff / (Math.sqrt(v[i] * vHatCoeff) + eps));
                }
            }
        }
    }

    @Override
    public void setLr(double lr) {
        // Note: This doesn't update the final lr field since it's immutable
        // For dynamic LR, use a scheduler instead
    }

    /**
     * Builder for Adam optimizer.
     */
    public static class Builder {
        private final List<GradTensor> parameters;
        private final double lr;
        private double beta1 = 0.9;
        private double beta2 = 0.999;
        private double eps = 1e-8;
        private double weightDecay = 0.0;
        private boolean amsgrad = false;

        private Builder(List<GradTensor> parameters, double lr) {
            this.parameters = parameters;
            this.lr = lr;
        }

        /**
         * Set beta coefficients for moment estimates.
         *
         * @param beta1 coefficient for first moment (default 0.9)
         * @param beta2 coefficient for second moment (default 0.999)
         * @return this builder
         */
        public Builder betas(double beta1, double beta2) {
            this.beta1 = beta1;
            this.beta2 = beta2;
            return this;
        }

        /**
         * Set epsilon for numerical stability.
         *
         * @param eps epsilon value (default 1e-8)
         * @return this builder
         */
        public Builder eps(double eps) {
            this.eps = eps;
            return this;
        }

        /**
         * Set weight decay (L2 regularization).
         *
         * @param weightDecay weight decay coefficient (default 0.0)
         * @return this builder
         */
        public Builder weightDecay(double weightDecay) {
            this.weightDecay = weightDecay;
            return this;
        }

        /**
         * Enable AMSGrad variant.
         *
         * @param enabled true to enable (default false)
         * @return this builder
         */
        public Builder amsgrad(boolean enabled) {
            this.amsgrad = enabled;
            return this;
        }

        /**
         * Build the Adam optimizer.
         *
         * @return configured optimizer
         */
        public Adam build() {
            return new Adam(this);
        }
    }
}
