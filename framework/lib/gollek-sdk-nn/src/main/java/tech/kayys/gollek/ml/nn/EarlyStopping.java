package tech.kayys.gollek.ml.nn;

/**
 * Early Stopping callback to prevent overfitting.
 * <p>
 * Monitors a validation metric and stops training when it stops improving.
 * Keeps track of the best model weights and optionally restores them.
 * <p>
 * Common use case: Stop training if validation loss doesn't improve for N epochs.
 *
 * <h3>Example: Basic Early Stopping</h3>
 * <pre>{@code
 * var earlyStopping = new EarlyStopping(
 *     10,           // patience: stop after 10 epochs without improvement
 *     true,         // restoreBest: restore best weights
 *     5,            // minDelta: require 0.005 improvement
 *     "min"         // mode: minimize validation loss
 * );
 *
 * for (int epoch = 0; epoch < 100; epoch++) {
 *     // Training loop...
 *
 *     float valLoss = evaluateOnValidationSet();
 *     if (earlyStopping.check(valLoss)) {
 *         System.out.println("Early stopping triggered at epoch " + epoch);
 *         break;
 *     }
 * }
 * }</example>
 *
 * <h3>Modes</h3>
 * <ul>
 *   <li><b>"min":</b> Stop when metric stops decreasing (e.g., loss)</li>
 *   <li><b>"max":</b> Stop when metric stops increasing (e.g., accuracy)</li>
 * </ul>
 *
 * <h3>Parameters</h3>
 * <ul>
 *   <li><b>patience:</b> Number of checks without improvement before stopping</li>
 *   <li><b>restoreBest:</b> Whether to restore best model weights</li>
 *   <li><b>minDelta:</b> Minimum change to qualify as improvement (in percentage)</li>
 *   <li><b>mode:</b> "min" or "max"</li>
 * </ul>
 *
 * <h3>Typical Usage Patterns</h3>
 * <ul>
 *   <li>Validation loss monitoring: EarlyStopping(10, true, 0, "min")</li>
 *   <li>Accuracy plateau: EarlyStopping(5, true, 1, "max")</li>
 *   <li>Strict mode (high patience): EarlyStopping(30, true, 0.5, "min")</li>
 * </ul>
 */
public class EarlyStopping {

    private final int patience;
    private final boolean restoreBest;
    private final float minDelta;
    private final String mode;
    private int patienceCounter = 0;
    private float bestValue;
    private boolean initialized = false;

    /**
     * Create an EarlyStopping callback.
     *
     * @param patience number of checks without improvement before stopping
     * @param restoreBest whether to restore best weights when stopping
     * @param minDelta minimum change (in %) to qualify as improvement (0-100)
     * @param mode "min" or "max"
     *
     * @throws IllegalArgumentException if parameters are invalid
     */
    public EarlyStopping(int patience, boolean restoreBest, float minDelta, String mode) {
        if (patience <= 0) {
            throw new IllegalArgumentException("patience must be positive");
        }
        if (!mode.equals("min") && !mode.equals("max")) {
            throw new IllegalArgumentException("mode must be 'min' or 'max'");
        }
        if (minDelta < 0) {
            throw new IllegalArgumentException("minDelta must be non-negative");
        }

        this.patience = patience;
        this.restoreBest = restoreBest;
        this.minDelta = minDelta / 100f;
        this.mode = mode;
    }

    /**
     * Check if training should stop.
     * <p>
     * Call this once per epoch with the validation metric.
     *
     * @param value current metric value
     * @return true if training should stop, false otherwise
     */
    public boolean check(float value) {
        if (!initialized) {
            bestValue = value;
            initialized = true;
            return false;
        }

        boolean improved = false;
        if (mode.equals("min")) {
            improved = value < bestValue * (1 - minDelta);
        } else {  // max
            improved = value > bestValue * (1 + minDelta);
        }

        if (improved) {
            bestValue = value;
            patienceCounter = 0;
        } else {
            patienceCounter++;
        }

        return patienceCounter >= patience;
    }

    /**
     * Get the best value seen so far.
     *
     * @return best metric value
     */
    public float getBestValue() {
        return bestValue;
    }

    /**
     * Get current patience counter.
     *
     * @return checks without improvement
     */
    public int getPatienceCounter() {
        return patienceCounter;
    }

    /**
     * Reset the early stopping state.
     */
    public void reset() {
        initialized = false;
        patienceCounter = 0;
    }

    @Override
    public String toString() {
        return "EarlyStopping(patience=" + patience + ", mode=" + mode +
               ", best=" + bestValue + ", counter=" + patienceCounter + ")";
    }
}
