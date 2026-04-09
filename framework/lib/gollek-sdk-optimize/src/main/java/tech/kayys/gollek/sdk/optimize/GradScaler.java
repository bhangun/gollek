package tech.kayys.gollek.sdk.optimize;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.List;

/**
 * Gradient scaler for mixed precision (FP16) training.
 *
 * <p>Implements dynamic loss scaling to prevent gradient underflow during
 * FP16 training. The scaler automatically adjusts the scale factor based
 * on overflow detection.</p>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * GradScaler scaler = new GradScaler()
 *     .initScale(65536.0)
 *     .growthInterval(2000)
 *     .build();
 *
 * // In training loop:
 * GradTensor loss = model.forward(inputs).mseLoss(targets);
 * scaler.scale(loss).backward();
 * scaler.unscaleAndClip(optimizer.getParameters(), maxNorm);
 * scaler.step(optimizer);
 * scaler.update();
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class GradScaler {

    private double scale;
    private final double growthFactor;
    private final double backoffFactor;
    private final int growthInterval;
    private final double minScale;
    private final double maxScale;

    private int consecutiveNonInfSteps = 0;
    private boolean overflowDetected = false;

    private GradScaler(Builder builder) {
        this.scale = builder.initScale;
        this.growthFactor = builder.growthFactor;
        this.backoffFactor = builder.backoffFactor;
        this.growthInterval = builder.growthInterval;
        this.minScale = builder.minScale;
        this.maxScale = builder.maxScale;
    }

    /**
     * Create a new GradScaler builder.
     *
     * @return builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create GradScaler with default settings.
     *
     * @return configured scaler
     */
    public static GradScaler create() {
        return builder().build();
    }

    /**
     * Scale a loss tensor before backward.
     *
     * @param loss original loss tensor
     * @return scaled loss tensor
     */
    public GradTensor scale(GradTensor loss) {
        float[] data = loss.data();
        for (int i = 0; i < data.length; i++) {
            data[i] *= (float) scale;
        }
        return loss;
    }

    /**
     * Unscale gradients and check for overflow.
     *
     * @param parameters list of parameters with gradients
     * @return true if overflow was detected
     */
    public boolean unscaleAndCheck(List<GradTensor> parameters) {
        overflowDetected = false;

        for (GradTensor param : parameters) {
            if (param.grad() == null) continue;

            float[] grad = param.grad().data();
            for (int i = 0; i < grad.length; i++) {
                float unscaled = grad[i] / (float) scale;
                grad[i] = unscaled;

                // Check for inf/NaN
                if (Float.isNaN(unscaled) || Float.isInfinite(unscaled)) {
                    overflowDetected = true;
                }
            }
        }

        return overflowDetected;
    }

    /**
     * Unscale gradients and clip by norm.
     *
     * @param parameters list of parameters with gradients
     * @param maxNorm    maximum gradient norm
     * @return actual total norm before clipping
     */
    public double unscaleAndClipByNorm(List<GradTensor> parameters, double maxNorm) {
        overflowDetected = false;
        double totalNorm = 0.0;

        // First pass: unscale and compute norm
        for (GradTensor param : parameters) {
            if (param.grad() == null) continue;

            float[] grad = param.grad().data();
            for (int i = 0; i < grad.length; i++) {
                float unscaled = grad[i] / (float) scale;
                grad[i] = unscaled;

                // Check for inf/NaN
                if (Float.isNaN(unscaled) || Float.isInfinite(unscaled)) {
                    overflowDetected = true;
                }

                totalNorm += unscaled * unscaled;
            }
        }

        totalNorm = Math.sqrt(totalNorm);

        // Second pass: clip if needed
        if (totalNorm > maxNorm) {
            double clipCoeff = maxNorm / (totalNorm + 1e-6);
            for (GradTensor param : parameters) {
                if (param.grad() == null) continue;

                float[] grad = param.grad().data();
                for (int i = 0; i < grad.length; i++) {
                    grad[i] *= (float) clipCoeff;
                }
            }
        }

        return totalNorm;
    }

    /**
     * Unscale gradients and clip by value.
     *
     * @param parameters list of parameters with gradients
     * @param minVal     minimum gradient value
     * @param maxVal     maximum gradient value
     */
    public void unscaleAndClipByValue(List<GradTensor> parameters, float minVal, float maxVal) {
        overflowDetected = false;

        for (GradTensor param : parameters) {
            if (param.grad() == null) continue;

            float[] grad = param.grad().data();
            for (int i = 0; i < grad.length; i++) {
                float unscaled = grad[i] / (float) scale;
                grad[i] = unscaled;

                // Check for inf/NaN
                if (Float.isNaN(unscaled) || Float.isInfinite(unscaled)) {
                    overflowDetected = true;
                }

                // Clip
                grad[i] = Math.max(minVal, Math.min(maxVal, grad[i]));
            }
        }
    }

    /**
     * Perform optimizer step with scaled gradients.
     *
     * @param optimizer optimizer to step
     */
    public void step(Optimizer optimizer) {
        if (!overflowDetected) {
            optimizer.step();
        }
    }

    /**
     * Update the scale factor based on overflow detection.
     *
     * <p>If overflow was detected, decrease the scale. If no overflow for
     * growthInterval steps, increase the scale.</p>
     */
    public void update() {
        if (overflowDetected) {
            scale = Math.max(minScale, scale * backoffFactor);
            consecutiveNonInfSteps = 0;
            overflowDetected = false;
        } else {
            consecutiveNonInfSteps++;
            if (consecutiveNonInfSteps >= growthInterval) {
                scale = Math.min(maxScale, scale * growthFactor);
                consecutiveNonInfSteps = 0;
            }
        }
    }

    /**
     * Get the current scale factor.
     *
     * @return current scale
     */
    public double getScale() {
        return scale;
    }

    /**
     * Check if overflow was detected in the last unscale operation.
     *
     * @return true if overflow detected
     */
    public boolean isOverflowDetected() {
        return overflowDetected;
    }

    /**
     * Get the number of consecutive non-overflow steps.
     *
     * @return consecutive non-overflow steps
     */
    public int getConsecutiveNonInfSteps() {
        return consecutiveNonInfSteps;
    }

    /**
     * Builder for GradScaler.
     */
    public static class Builder {
        private double initScale = 65536.0;
        private double growthFactor = 2.0;
        private double backoffFactor = 0.5;
        private int growthInterval = 2000;
        private double minScale = 1.0;
        private double maxScale = 1e10;

        private Builder() {}

        /**
         * Set initial scale factor.
         *
         * @param scale initial scale (default 65536.0)
         * @return this builder
         */
        public Builder initScale(double scale) {
            this.initScale = scale;
            return this;
        }

        /**
         * Set growth factor for scale increase.
         *
         * @param factor growth factor (default 2.0)
         * @return this builder
         */
        public Builder growthFactor(double factor) {
            this.growthFactor = factor;
            return this;
        }

        /**
         * Set backoff factor for scale decrease on overflow.
         *
         * @param factor backoff factor (default 0.5)
         * @return this builder
         */
        public Builder backoffFactor(double factor) {
            this.backoffFactor = factor;
            return this;
        }

        /**
         * Set growth interval (steps between scale increases).
         *
         * @param interval growth interval (default 2000)
         * @return this builder
         */
        public Builder growthInterval(int interval) {
            this.growthInterval = interval;
            return this;
        }

        /**
         * Set minimum allowed scale.
         *
         * @param minScale minimum scale (default 1.0)
         * @return this builder
         */
        public Builder minScale(double minScale) {
            this.minScale = minScale;
            return this;
        }

        /**
         * Set maximum allowed scale.
         *
         * @param maxScale maximum scale (default 1e10)
         * @return this builder
         */
        public Builder maxScale(double maxScale) {
            this.maxScale = maxScale;
            return this;
        }

        /**
         * Build the GradScaler instance.
         *
         * @return configured scaler
         */
        public GradScaler build() {
            return new GradScaler(this);
        }
    }
}
