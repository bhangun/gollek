package tech.kayys.gollek.sdk.train;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.DoubleSupplier;

/**
 * Early stopping callback.
 *
 * <p>Stops training when the monitored metric stops improving for a specified
 * number of epochs (patience).</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Callback earlyStop = EarlyStopping.builder()
 *     .patience(5)
 *     .minDelta(0.001)
 *     .mode(Mode.MIN)  // Monitor validation loss (lower is better)
 *     .build();
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class EarlyStopping implements Callback {

    private static final Logger log = LoggerFactory.getLogger(EarlyStopping.class);

    /**
     * Monitoring mode.
     */
    public enum Mode {
        /** Metric should decrease (e.g., loss) */
        MIN,
        /** Metric should increase (e.g., accuracy) */
        MAX
    }

    private final int patience;
    private final double minDelta;
    private final Mode mode;
    private final String monitor;

    private int epochsNoImprove = 0;
    private double bestValue;
    private boolean stopped = false;

    private EarlyStopping(Builder builder) {
        this.patience = builder.patience;
        this.minDelta = builder.minDelta;
        this.mode = builder.mode;
        this.monitor = builder.monitor;
        this.bestValue = mode == Mode.MAX ? -Double.MAX_VALUE : Double.MAX_VALUE;
    }

    /**
     * Create a new builder.
     *
     * @return builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create early stopping with default settings.
     *
     * @param patience number of epochs to wait before stopping
     * @return configured callback
     */
    public static EarlyStopping patience(int patience) {
        return builder().patience(patience).build();
    }

    @Override
    public void onValidationEnd(Trainer trainer, int epoch, double valLoss) {
        double currentValue = valLoss;

        boolean improved = mode == Mode.MIN
                ? currentValue < bestValue - minDelta
                : currentValue > bestValue + minDelta;

        if (improved) {
            bestValue = currentValue;
            epochsNoImprove = 0;
            log.debug("Epoch {}: {} improved to {:.6f}", epoch, monitor, currentValue);
        } else {
            epochsNoImprove++;
            log.debug("Epoch {}: {} did not improve ({}/{})", epoch, monitor, epochsNoImprove, patience);

            if (epochsNoImprove >= patience) {
                stopped = true;
                log.info("Early stopping triggered at epoch {} (best {}={:.6f})",
                        epoch, monitor, bestValue);
                trainer.stop();
            }
        }
    }

    /**
     * Check if early stopping should trigger.
     *
     * @return true if should stop
     */
    public boolean shouldStop() {
        return stopped;
    }

    /**
     * Get the best monitored value seen so far.
     *
     * @return best value
     */
    public double getBestValue() {
        return bestValue;
    }

    /**
     * Get number of epochs since last improvement.
     *
     * @return epochs without improvement
     */
    public int getEpochsNoImprove() {
        return epochsNoImprove;
    }

    /**
     * Builder for EarlyStopping.
     */
    public static class Builder {
        private int patience = 5;
        private double minDelta = 0.0;
        private Mode mode = Mode.MIN;
        private String monitor = "val_loss";

        private Builder() {}

        /**
         * Set patience (epochs to wait before stopping).
         *
         * @param patience patience value
         * @return this builder
         */
        public Builder patience(int patience) {
            this.patience = patience;
            return this;
        }

        /**
         * Set minimum change in monitored value to qualify as improvement.
         *
         * @param minDelta minimum delta
         * @return this builder
         */
        public Builder minDelta(double minDelta) {
            this.minDelta = minDelta;
            return this;
        }

        /**
         * Set monitoring mode.
         *
         * @param mode MIN for loss, MAX for accuracy
         * @return this builder
         */
        public Builder mode(Mode mode) {
            this.mode = mode;
            return this;
        }

        /**
         * Set metric name to monitor.
         *
         * @param monitor metric name
         * @return this builder
         */
        public Builder monitor(String monitor) {
            this.monitor = monitor;
            return this;
        }

        /**
         * Build the EarlyStopping instance.
         *
         * @return configured callback
         */
        public EarlyStopping build() {
            return new EarlyStopping(this);
        }
    }
}
