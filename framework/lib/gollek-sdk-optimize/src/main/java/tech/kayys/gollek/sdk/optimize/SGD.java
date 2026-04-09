package tech.kayys.gollek.sdk.optimize;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.List;

/**
 * Stochastic Gradient Descent optimizer with momentum and Nesterov acceleration.
 *
 * <p>Implements SGD with optional momentum and Nesterov accelerated gradient.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Optimizer optimizer = SGD.builder(model.parameters(), 0.01)
 *     .momentum(0.9)
 *     .nesterov(true)
 *     .weightDecay(0.0001)
 *     .build();
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class SGD extends Optimizer {

    private final double momentum;
    private final double weightDecay;
    private final boolean nesterov;
    private double currentLr;

    // Per-parameter momentum buffer
    private final java.util.Map<GradTensor, float[]> momentumBuffers = new java.util.HashMap<>();

    private SGD(Builder builder) {
        super(builder.parameters, builder.lr);
        this.momentum = builder.momentum;
        this.weightDecay = builder.weightDecay;
        this.nesterov = builder.nesterov;
        this.currentLr = builder.lr;
    }

    /**
     * Create a new SGD builder.
     *
     * @param parameters list of parameters to optimize
     * @param lr         learning rate
     * @return builder instance
     */
    public static Builder builder(List<GradTensor> parameters, double lr) {
        return new Builder(parameters, lr);
    }

    /**
     * Create SGD with default settings.
     *
     * @param parameters list of parameters to optimize
     * @param lr         learning rate
     * @return optimizer instance
     */
    public static SGD create(List<GradTensor> parameters, double lr) {
        return builder(parameters, lr).build();
    }

    @Override
    public void step() {
        step++;

        for (GradTensor param : parameters) {
            if (param.grad() == null) continue;

            float[] grad = param.grad().data();
            float[] data = param.data();

            // Weight decay
            if (weightDecay != 0) {
                for (int i = 0; i < data.length; i++) {
                    grad[i] += weightDecay * data[i];
                }
            }

            // Momentum
            if (momentum > 0) {
                if (!momentumBuffers.containsKey(param)) {
                    momentumBuffers.put(param, new float[data.length]);
                }

                float[] buf = momentumBuffers.get(param);

                if (nesterov) {
                    // Nesterov accelerated gradient
                    for (int i = 0; i < data.length; i++) {
                        buf[i] = (float) (momentum * buf[i] + grad[i]);
                        data[i] -= (float) (currentLr * (grad[i] + momentum * buf[i]));
                    }
                } else {
                    // Standard momentum
                    for (int i = 0; i < data.length; i++) {
                        buf[i] = (float) (momentum * buf[i] + grad[i]);
                        data[i] -= (float) (currentLr * buf[i]);
                    }
                }
            } else {
                // Plain SGD
                for (int i = 0; i < data.length; i++) {
                    data[i] -= currentLr * grad[i];
                }
            }
        }
    }

    @Override
    public void setLr(double lr) {
        this.currentLr = lr;
    }

    @Override
    public double getLr() {
        return currentLr;
    }

    /**
     * Builder for SGD optimizer.
     */
    public static class Builder {
        private final List<GradTensor> parameters;
        private final double lr;
        private double momentum = 0.0;
        private double weightDecay = 0.0;
        private boolean nesterov = false;

        private Builder(List<GradTensor> parameters, double lr) {
            this.parameters = parameters;
            this.lr = lr;
        }

        /**
         * Set momentum factor.
         *
         * @param momentum momentum factor (default 0.0)
         * @return this builder
         */
        public Builder momentum(double momentum) {
            this.momentum = momentum;
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
         * Enable Nesterov accelerated gradient.
         *
         * @param nesterov true to enable (default false)
         * @return this builder
         */
        public Builder nesterov(boolean nesterov) {
            this.nesterov = nesterov;
            return this;
        }

        /**
         * Build the SGD optimizer.
         *
         * @return configured optimizer
         */
        public SGD build() {
            return new SGD(this);
        }
    }
}
