package tech.kayys.gollek.sdk.train;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Cosine annealing learning rate scheduler.
 *
 * <p>Sets the learning rate using a cosine annealing schedule:</p>
 * <pre>
 * lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * epoch / T_max))
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * LRScheduler scheduler = CosineAnnealingLR.builder()
 *     .maxLr(0.001)
 *     .minLr(0.00001)
 *     .tMax(100)  // Total epochs
 *     .build();
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class CosineAnnealingLR implements LRScheduler {

    private static final Logger log = LoggerFactory.getLogger(CosineAnnealingLR.class);

    private final double maxLr;
    private final double minLr;
    private final int tMax;

    private int lastEpoch = -1;
    private double currentLr;

    private CosineAnnealingLR(Builder builder) {
        this.maxLr = builder.maxLr;
        this.minLr = builder.minLr;
        this.tMax = builder.tMax;
        this.currentLr = maxLr;
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

        if (lastEpoch < 0) {
            currentLr = maxLr;
        } else if (lastEpoch >= tMax) {
            currentLr = minLr;
        } else {
            currentLr = minLr + 0.5 * (maxLr - minLr) *
                    (1.0 + Math.cos(Math.PI * lastEpoch / tMax));
        }
    }

    @Override
    public double getCurrentLr() {
        return currentLr;
    }

    @Override
    public double getBaseLr() {
        return maxLr;
    }

    @Override
    public void reset() {
        lastEpoch = -1;
        currentLr = maxLr;
    }

    /**
     * Get maximum learning rate.
     *
     * @return max LR
     */
    public double getMaxLr() {
        return maxLr;
    }

    /**
     * Get minimum learning rate.
     *
     * @return min LR
     */
    public double getMinLr() {
        return minLr;
    }

    /**
     * Get total number of epochs for the annealing cycle.
     *
     * @return T_max
     */
    public int getTMax() {
        return tMax;
    }

    /**
     * Builder for CosineAnnealingLR.
     */
    public static class Builder {
        private double maxLr = 0.001;
        private double minLr = 0.0;
        private int tMax = 100;

        private Builder() {}

        /**
         * Set maximum learning rate.
         *
         * @param lr max learning rate
         * @return this builder
         */
        public Builder maxLr(double lr) {
            this.maxLr = lr;
            return this;
        }

        /**
         * Set minimum learning rate.
         *
         * @param lr min learning rate
         * @return this builder
         */
        public Builder minLr(double lr) {
            this.minLr = lr;
            return this;
        }

        /**
         * Set total number of epochs for annealing.
         *
         * @param tMax total epochs
         * @return this builder
         */
        public Builder tMax(int tMax) {
            this.tMax = tMax;
            return this;
        }

        /**
         * Build the CosineAnnealingLR instance.
         *
         * @return configured scheduler
         */
        public CosineAnnealingLR build() {
            return new CosineAnnealingLR(this);
        }
    }
}
