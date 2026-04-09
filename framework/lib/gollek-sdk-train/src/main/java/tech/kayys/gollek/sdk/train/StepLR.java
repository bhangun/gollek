package tech.kayys.gollek.sdk.train;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Step learning rate scheduler.
 *
 * <p>Decays the learning rate by gamma every stepSize epochs.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * LRScheduler scheduler = StepLR.builder()
 *     .baseLr(0.001)
 *     .stepSize(30)      // Decay every 30 epochs
 *     .gamma(0.1)        // Multiply LR by 0.1
 *     .build();
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class StepLR implements LRScheduler {

    private static final Logger log = LoggerFactory.getLogger(StepLR.class);

    private final double baseLr;
    private final int stepSize;
    private final double gamma;

    private int lastEpoch = -1;
    private double currentLr;

    private StepLR(Builder builder) {
        this.baseLr = builder.baseLr;
        this.stepSize = builder.stepSize;
        this.gamma = builder.gamma;
        this.currentLr = baseLr;
    }

    /**
     * Create a new builder.
     *
     * @return builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    @Override
    public void step(TrainingMetrics metrics) {
        lastEpoch++;

        if (lastEpoch > 0 && lastEpoch % stepSize == 0) {
            double oldLr = currentLr;
            currentLr = currentLr * gamma;
            log.debug("StepLR: decaying LR from {:.6f} to {:.6f} (epoch {})", oldLr, currentLr, lastEpoch);
        }
    }

    @Override
    public double getCurrentLr() {
        return currentLr;
    }

    @Override
    public double getBaseLr() {
        return baseLr;
    }

    @Override
    public void reset() {
        lastEpoch = -1;
        currentLr = baseLr;
    }

    /**
     * Builder for StepLR.
     */
    public static class Builder {
        private double baseLr = 0.001;
        private int stepSize = 30;
        private double gamma = 0.1;

        private Builder() {}

        /**
         * Set base learning rate.
         *
         * @param lr learning rate
         * @return this builder
         */
        public Builder baseLr(double lr) {
            this.baseLr = lr;
            return this;
        }

        /**
         * Set step size (epochs between decays).
         *
         * @param stepSize step size
         * @return this builder
         */
        public Builder stepSize(int stepSize) {
            this.stepSize = stepSize;
            return this;
        }

        /**
         * Set decay factor (gamma).
         *
         * @param gamma decay factor (0 < gamma < 1)
         * @return this builder
         */
        public Builder gamma(double gamma) {
            this.gamma = gamma;
            return this;
        }

        /**
         * Build the StepLR instance.
         *
         * @return configured scheduler
         */
        public StepLR build() {
            return new StepLR(this);
        }
    }
}
