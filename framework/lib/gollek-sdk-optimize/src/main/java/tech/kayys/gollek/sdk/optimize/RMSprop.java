package tech.kayys.gollek.sdk.optimize;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.List;
import java.util.HashMap;
import java.util.Map;

/**
 * RMSprop optimizer.
 *
 * <p>Implements RMSProp from Geoff Hinton's lecture notes. Divides the learning rate
 * by an exponentially decaying average of squared gradients.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Optimizer optimizer = RMSprop.builder(model.parameters(), 0.01)
 *     .alpha(0.99)
 *     .eps(1e-8)
 *     .momentum(0.0)
 *     .build();
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class RMSprop extends Optimizer {

    private final double alpha;
    private final double eps;
    private final double momentum;
    private final double weightDecay;
    private double currentLr;

    // Per-parameter state
    private final Map<GradTensor, float[]> squareAvg = new HashMap<>();
    private final Map<GradTensor, float[]> momentumBuffers = new HashMap<>();

    private RMSprop(Builder builder) {
        super(builder.parameters, builder.lr);
        this.alpha = builder.alpha;
        this.eps = builder.eps;
        this.momentum = builder.momentum;
        this.weightDecay = builder.weightDecay;
        this.currentLr = builder.lr;
    }

    /**
     * Create a new RMSprop builder.
     *
     * @param parameters list of parameters to optimize
     * @param lr         learning rate
     * @return builder instance
     */
    public static Builder builder(List<GradTensor> parameters, double lr) {
        return new Builder(parameters, lr);
    }

    /**
     * Create RMSprop with default settings.
     *
     * @param parameters list of parameters to optimize
     * @param lr         learning rate
     * @return optimizer instance
     */
    public static RMSprop create(List<GradTensor> parameters, double lr) {
        return builder(parameters, lr).build();
    }

    @Override
    public void step() {
        step++;

        for (GradTensor param : parameters) {
            if (param.grad() == null) continue;

            float[] grad = param.grad().data();
            float[] data = param.data();

            // Initialize state if needed
            if (!squareAvg.containsKey(param)) {
                squareAvg.put(param, new float[data.length]);
                if (momentum > 0) {
                    momentumBuffers.put(param, new float[data.length]);
                }
            }

            float[] v = squareAvg.get(param);

            // Weight decay
            if (weightDecay != 0) {
                for (int i = 0; i < data.length; i++) {
                    grad[i] += weightDecay * data[i];
                }
            }

            // Update squared average
            for (int i = 0; i < data.length; i++) {
                v[i] = (float) (alpha * v[i] + (1.0 - alpha) * grad[i] * grad[i]);
            }

            // Update parameters
            if (momentum > 0) {
                float[] buf = momentumBuffers.get(param);
                for (int i = 0; i < data.length; i++) {
                    buf[i] = (float) (momentum * buf[i] + grad[i] / (Math.sqrt(v[i]) + eps));
                    data[i] -= (float) (currentLr * buf[i]);
                }
            } else {
                for (int i = 0; i < data.length; i++) {
                    data[i] -= (float) (currentLr * grad[i] / (Math.sqrt(v[i]) + eps));
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
     * Builder for RMSprop optimizer.
     */
    public static class Builder {
        private final List<GradTensor> parameters;
        private final double lr;
        private double alpha = 0.99;
        private double eps = 1e-8;
        private double momentum = 0.0;
        private double weightDecay = 0.0;

        private Builder(List<GradTensor> parameters, double lr) {
            this.parameters = parameters;
            this.lr = lr;
        }

        /**
         * Set smoothing constant alpha.
         *
         * @param alpha alpha value (default 0.99)
         * @return this builder
         */
        public Builder alpha(double alpha) {
            this.alpha = alpha;
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
         * Build the RMSprop optimizer.
         *
         * @return configured optimizer
         */
        public RMSprop build() {
            return new RMSprop(this);
        }
    }
}
