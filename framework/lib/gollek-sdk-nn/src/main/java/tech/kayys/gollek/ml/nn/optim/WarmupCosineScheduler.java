package tech.kayys.gollek.ml.nn.optim;

/**
 * Linear warmup followed by cosine annealing learning rate schedule.
 *
 * <p>Used widely in Transformer fine-tuning (BERT, GPT, LLaMA).
 * During the warmup phase the learning rate increases linearly from 0 to
 * {@code maxLr}; after warmup it follows a cosine decay down to {@code minLr}.
 *
 * <pre>
 *   step &lt; warmupSteps:  lr = maxLr * step / warmupSteps
 *   step &ge; warmupSteps: lr = minLr + 0.5*(maxLr-minLr)*(1 + cos(π*(step-warmup)/(total-warmup)))
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var scheduler = new WarmupCosineScheduler(optimizer,
 *     warmupSteps = 100, totalSteps = 1000,
 *     maxLr = 3e-4f, minLr = 1e-6f);
 *
 * for (int step = 0; step < 1000; step++) {
 *     optimizer.step();
 *     scheduler.step();
 * }
 * }</pre>
 */
public final class WarmupCosineScheduler extends LRScheduler {

    private final int   warmupSteps;
    private final int   totalSteps;
    private final float maxLr;
    private final float minLr;
    private int   currentStep = 0;
    private float currentLr;

    /**
     * Constructs a warmup-cosine scheduler.
     *
     * @param optimizer    the optimizer whose learning rate will be updated
     * @param warmupSteps  number of linear warmup steps
     * @param totalSteps   total training steps (warmup + cosine decay)
     * @param maxLr        peak learning rate (reached at end of warmup)
     * @param minLr        minimum learning rate (reached at end of cosine decay)
     */
    public WarmupCosineScheduler(Optimizer optimizer, int warmupSteps, int totalSteps,
                                  float maxLr, float minLr) {
        super(optimizer);
        this.warmupSteps = warmupSteps;
        this.totalSteps  = totalSteps;
        this.maxLr       = maxLr;
        this.minLr       = minLr;
        this.currentLr   = 0f; // starts at 0
    }

    /**
     * Advances the scheduler by one step and updates the optimizer's learning rate.
     *
     * <p>Call this <em>after</em> {@code optimizer.step()} each training step.
     */
    @Override
    public void step() {
        currentStep++;
        if (currentStep <= warmupSteps) {
            currentLr = maxLr * (float) currentStep / warmupSteps;
        } else {
            float progress = (float) (currentStep - warmupSteps) / (totalSteps - warmupSteps);
            currentLr = minLr + 0.5f * (maxLr - minLr)
                        * (1f + (float) Math.cos(Math.PI * progress));
        }
        setLearningRate(currentLr);
    }

    /**
     * Returns the current learning rate.
     *
     * @return current lr value
     */
    @Override
    public float getLr() { return currentLr; }

    /** @return current step count */
    public int currentStep() { return currentStep; }
}
