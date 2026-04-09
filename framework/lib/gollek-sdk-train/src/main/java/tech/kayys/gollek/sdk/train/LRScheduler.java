package tech.kayys.gollek.sdk.train;

/**
 * Learning rate scheduler interface.
 *
 * <p>Implementations adjust the learning rate during training based on various
 * strategies (step decay, cosine annealing, reduce on plateau, etc.).</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * LRScheduler scheduler = StepLR.builder()
 *     .baseLr(0.001)
 *     .stepSize(30)
 *     .gamma(0.1)
 *     .build();
 *
 * trainer.scheduler(scheduler);
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public interface LRScheduler {

    /**
     * Update learning rate based on current training state.
     *
     * @param metrics current training metrics
     */
    void step(TrainingMetrics metrics);

    /**
     * Get the current learning rate.
     *
     * @return current learning rate
     */
    double getCurrentLr();

    /**
     * Get the base (initial) learning rate.
     *
     * @return base learning rate
     */
    double getBaseLr();

    /**
     * Reset the scheduler to initial state.
     */
    void reset();
}
