package tech.kayys.gollek.ml.nn.optim;

/**
 * Cosine Annealing Learning Rate Scheduler.
 * <p>
 * Uses cosine annealing to smoothly decay the learning rate from initial value to a minimum value.
 * This often provides better convergence than step-based schedules, especially for long training runs.
 * <p>
 * {@code lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * step / T_max))}
 *
 * <h3>Example: Cosine annealing over 100 epochs</h3>
 * <pre>{@code
 * var scheduler = new CosineAnnealingLR(optimizer, 100, 1e-6f);
 *
 * for (int epoch = 0; epoch < 100; epoch++) {
 *     // Training...
 *     scheduler.step();
 * }
 * // Learning rate smoothly decays from initial_lr to 1e-6
 * // following a cosine curve
 * }</pre>
 *
 * <h3>Key Features</h3>
 * <ul>
 *   <li>Smooth decay (not step-wise)</li>
 *   <li>Often provides better generalization</li>
 *   <li>Popular in modern deep learning</li>
 *   <li>Allows for warm restarts with periodic schedule</li>
 * </ul>
 *
 * <h3>Typical Usage</h3>
 * <ul>
 *   <li>Total epochs known in advance</li>
 *   <li>Works well with SGD and Adam</li>
 *   <li>Common in vision (ResNet) and NLP (BERT fine-tuning)</li>
 * </ul>
 *
 * @see StepLR
 * @see ExponentialLR
 */
public class CosineAnnealingLR extends LRScheduler {

    private final float initialLr;
    private final int tMax;
    private final float minLr;
    private int step = 0;

    /**
     * Create a CosineAnnealingLR scheduler.
     *
     * @param optimizer the optimizer to schedule
     * @param tMax      maximum number of iterations (e.g., total epochs)
     * @param minLr     minimum learning rate at the end
     *
     * @throws IllegalArgumentException if tMax <= 0 or minLr < 0
     */
    public CosineAnnealingLR(Optimizer optimizer, int tMax, float minLr) {
        super(optimizer);
        if (tMax <= 0) {
            throw new IllegalArgumentException("tMax must be positive, got: " + tMax);
        }
        if (minLr < 0) {
            throw new IllegalArgumentException("minLr must be non-negative, got: " + minLr);
        }
        this.initialLr = optimizer.learningRate();
        this.tMax = tMax;
        this.minLr = minLr;
    }

    /**
     * Update learning rate for this step using cosine annealing.
     */
    @Override
    public void step() {
        if (step >= tMax) {
            // Keep at minimum after reaching tMax
            setLearningRate(minLr);
        } else {
            // Cosine annealing formula
            float cosineDecay = (float) Math.cos(Math.PI * step / tMax);
            float newLr = minLr + 0.5f * (initialLr - minLr) * (1 + cosineDecay);
            setLearningRate(newLr);
        }
        step++;
    }

    @Override
    public float getLr() {
        return optimizer.learningRate();
    }

    @Override
    public String toString() {
        return "CosineAnnealingLR(tMax=" + tMax + ", minLr=" + minLr + ")";
    }
}
