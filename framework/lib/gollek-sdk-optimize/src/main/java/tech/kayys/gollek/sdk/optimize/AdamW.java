package tech.kayys.gollek.sdk.optimize;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.List;
import java.util.HashMap;
import java.util.Map;

/**
 * AdamW optimizer with decoupled weight decay.
 *
 * <p>Implements AdamW from "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017).
 * Unlike Adam with L2 regularization, AdamW applies weight decay directly to the parameters
 * rather than to the gradients, leading to better generalization.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Optimizer optimizer = AdamW.builder(model.parameters(), 0.001)
 *     .betas(0.9, 0.999)
 *     .weightDecay(0.01)
 *     .build();
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class AdamW extends Optimizer {

    private final double beta1;
    private final double beta2;
    private final double eps;
    private final double weightDecay;

    // Per-parameter state
    private final Map<GradTensor, float[]> expAvg = new HashMap<>();
    private final Map<GradTensor, float[]> expAvgSq = new HashMap<>();

    private AdamW(Builder builder) {
        super(builder.parameters, builder.lr);
        this.beta1 = builder.beta1;
        this.beta2 = builder.beta2;
        this.eps = builder.eps;
        this.weightDecay = builder.weightDecay;
    }

    /**
     * Create a new AdamW builder.
     *
     * @param parameters list of parameters to optimize
     * @param lr         learning rate
     * @return builder instance
     */
    public static Builder builder(List<GradTensor> parameters, double lr) {
        return new Builder(parameters, lr);
    }

    /**
     * Create AdamW with default settings.
     *
     * @param parameters list of parameters to optimize
     * @param lr         learning rate
     * @return optimizer instance
     */
    public static AdamW create(List<GradTensor> parameters, double lr) {
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
            }

            float[] m = expAvg.get(param);
            float[] v = expAvg.get(param);

            // Decoupled weight decay (applied directly to parameters)
            if (weightDecay != 0) {
                for (int i = 0; i < data.length; i++) {
                    data[i] *= (1.0 - lr * weightDecay);
                }
            }

            // Update biased first and second moment estimates
            for (int i = 0; i < data.length; i++) {
                m[i] = (float) (beta1 * m[i] + (1.0 - beta1) * grad[i]);
                v[i] = (float) (beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i]);
            }

            // Bias-corrected estimates and parameter update
            float mHatCoeff = (float) (1.0 / biasCorrection1);
            float vHatCoeff = (float) (1.0 / biasCorrection2);

            for (int i = 0; i < data.length; i++) {
                data[i] -= (float) (lr * m[i] * mHatCoeff / (Math.sqrt(v[i] * vHatCoeff) + eps));
            }
        }
    }

    @Override
    public void setLr(double lr) {
        // Note: This doesn't update the final lr field since it's immutable
        // For dynamic LR, use a scheduler instead
    }

    /**
     * Builder for AdamW optimizer.
     */
    public static class Builder {
        private final List<GradTensor> parameters;
        private final double lr;
        private double beta1 = 0.9;
        private double beta2 = 0.999;
        private double eps = 1e-8;
        private double weightDecay = 0.01;

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
         * Set weight decay (decoupled from gradients).
         *
         * @param weightDecay weight decay coefficient (default 0.01)
         * @return this builder
         */
        public Builder weightDecay(double weightDecay) {
            this.weightDecay = weightDecay;
            return this;
        }

        /**
         * Build the AdamW optimizer.
         *
         * @return configured optimizer
         */
        public AdamW build() {
            return new AdamW(this);
        }
    }
}
