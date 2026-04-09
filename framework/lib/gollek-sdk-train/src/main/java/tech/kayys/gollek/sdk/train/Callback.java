package tech.kayys.gollek.sdk.train;

import java.io.Closeable;

/**
 * Callback interface for training events.
 *
 * <p>Implement this interface to hook into various points of the training loop
 * for logging, checkpointing, early stopping, or custom behavior.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * public class MyCallback implements Callback {
 *     @Override
 *     public void onEpochEnd(Trainer trainer, int epoch, double trainLoss) {
 *         System.out.printf("Epoch %d: loss=%.4f%n", epoch, trainLoss);
 *     }
 * }
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public interface Callback extends Closeable {

    /**
     * Called when training starts.
     *
     * @param trainer trainer instance
     */
    default void onTrainingStart(Trainer trainer) {}

    /**
     * Called at the start of each epoch.
     *
     * @param trainer trainer instance
     * @param epoch   epoch number (0-based)
     */
    default void onEpochStart(Trainer trainer, int epoch) {}

    /**
     * Called at the end of each epoch (after training, before validation).
     *
     * @param trainer    trainer instance
     * @param epoch      epoch number (0-based)
     * @param trainLoss  average training loss for the epoch
     */
    default void onEpochEnd(Trainer trainer, int epoch, double trainLoss) {}

    /**
     * Called after validation completes.
     *
     * @param trainer  trainer instance
     * @param epoch    epoch number (0-based)
     * @param valLoss  average validation loss for the epoch
     */
    default void onValidationEnd(Trainer trainer, int epoch, double valLoss) {}

    /**
     * Called at the start of each batch.
     *
     * @param trainer trainer instance
     * @param step    global step number
     */
    default void onBatchStart(Trainer trainer, int step) {}

    /**
     * Called at the end of each batch.
     *
     * @param trainer trainer instance
     * @param step    global step number
     * @param loss    loss value for this batch
     */
    default void onBatchEnd(Trainer trainer, int step, double loss) {}

    /**
     * Called when early stopping is triggered.
     *
     * @param trainer trainer instance
     * @param epoch   epoch number when stopped
     */
    default void onEarlyStopping(Trainer trainer, int epoch) {}

    /**
     * Called when a training error occurs.
     *
     * @param trainer trainer instance
     * @param error   the exception that occurred
     */
    default void onTrainingError(Trainer trainer, Exception error) {}

    /**
     * Called when training ends (successfully or via early stopping).
     *
     * @param trainer trainer instance
     */
    default void onTrainingEnd(Trainer trainer) {}

    @Override
    default void close() {
        // Default: no-op
    }
}
