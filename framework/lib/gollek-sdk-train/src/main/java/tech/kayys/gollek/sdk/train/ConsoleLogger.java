package tech.kayys.gollek.sdk.train;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Console logging callback.
 *
 * <p>Prints training progress to console with configurable verbosity.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Callback logger = ConsoleLogger.builder()
 *     .logInterval(10)  // Log every 10 batches
 *     .showProgressBar(true)
 *     .build();
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class ConsoleLogger implements Callback {

    private static final Logger log = LoggerFactory.getLogger(ConsoleLogger.class);

    private final int logInterval;
    private final boolean showProgressBar;
    private final boolean showMetrics;

    private long epochStartTime;
    private int batchCount;

    private ConsoleLogger(Builder builder) {
        this.logInterval = builder.logInterval;
        this.showProgressBar = builder.showProgressBar;
        this.showMetrics = builder.showMetrics;
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
     * Create console logger with default settings.
     *
     * @return configured callback
     */
    public static ConsoleLogger create() {
        return builder().build();
    }

    @Override
    public void onEpochStart(Trainer trainer, int epoch) {
        epochStartTime = System.currentTimeMillis();
        batchCount = 0;
        log.info("Epoch {}/{} started", epoch + 1, trainer.getConfig().epochs());
    }

    @Override
    public void onBatchEnd(Trainer trainer, int step, double loss) {
        batchCount++;

        if (batchCount % logInterval == 0) {
            log.debug("  Step {}: loss={:.6f}", step, loss);
        }
    }

    @Override
    public void onEpochEnd(Trainer trainer, int epoch, double trainLoss) {
        long duration = System.currentTimeMillis() - epochStartTime;
        log.info("Epoch {}/{} completed - train_loss={:.6f}, time={}ms",
                epoch + 1, trainer.getConfig().epochs(), trainLoss, duration);
    }

    @Override
    public void onValidationEnd(Trainer trainer, int epoch, double valLoss) {
        log.info("  Validation: val_loss={:.6f}", valLoss);

        if (showMetrics) {
            TrainingMetrics metrics = trainer.getMetrics();
            log.info("  Best val_loss: {:.6f} (epoch {})",
                    metrics.getBestValLoss(), metrics.getBestValLossEpoch() + 1);
        }
    }

    @Override
    public void onEarlyStopping(Trainer trainer, int epoch) {
        log.info("Early stopping triggered at epoch {}", epoch + 1);
    }

    @Override
    public void onTrainingEnd(Trainer trainer) {
        TrainingMetrics metrics = trainer.getMetrics();
        log.info("Training completed: {} epochs, best val_loss={:.6f}, duration={:.1f}s",
                metrics.getEpochCount(),
                metrics.getBestValLoss(),
                metrics.getDurationMs() / 1000.0);
    }

    @Override
    public void onTrainingError(Trainer trainer, Exception error) {
        log.error("Training failed: {}", error.getMessage(), error);
    }

    /**
     * Builder for ConsoleLogger.
     */
    public static class Builder {
        private int logInterval = 10;
        private boolean showProgressBar = false;
        private boolean showMetrics = true;

        private Builder() {}

        /**
         * Set logging interval (batches between logs).
         *
         * @param interval log interval
         * @return this builder
         */
        public Builder logInterval(int interval) {
            this.logInterval = interval;
            return this;
        }

        /**
         * Enable/disable progress bar.
         *
         * @param show true to show
         * @return this builder
         */
        public Builder showProgressBar(boolean show) {
            this.showProgressBar = show;
            return this;
        }

        /**
         * Enable/disable metrics display.
         *
         * @param show true to show
         * @return this builder
         */
        public Builder showMetrics(boolean show) {
            this.showMetrics = show;
            return this;
        }

        /**
         * Build the ConsoleLogger instance.
         *
         * @return configured callback
         */
        public ConsoleLogger build() {
            return new ConsoleLogger(this);
        }
    }
}
