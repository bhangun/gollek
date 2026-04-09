package tech.kayys.gollek.ml.nn.metrics;

/**
 * Accuracy Metric for classification.
 * <p>
 * Computes the percentage of correct predictions. Supports both multi-class
 * classification (argmax comparison) and binary classification.
 * <p>
 * {@code accuracy = (correct predictions) / (total predictions)}
 *
 * <h3>Example: Multi-class Classification</h3>
 * <pre>{@code
 * var metric = new Accuracy();
 *
 * // During training loop
 * for (var batch : dataloader) {
 *     var predictions = model.forward(batch.x);  // shape: [batch, num_classes]
 *     var targets = batch.y;                      // shape: [batch]
 *     metric.update(predictions, targets);
 * }
 *
 * float accuracy = metric.compute();  // between 0 and 1
 * System.out.println("Accuracy: " + (accuracy * 100) + "%");
 * }</example>
 *
 * <h3>Binary Classification</h3>
 * <pre>{@code
 * // For binary classification, use threshold of 0.5
 * var predictions = model.forward(x);  // shape: [batch, 1]
 * var targets = y;                      // shape: [batch]
 * metric.update(predictions, targets);
 * }</example>
 *
 * <h3>Multi-class Classification</h3>
 * For multi-class (C>2 classes):
 * <ul>
 *   <li>Predictions shape: [batch, num_classes] (logits or probabilities)</li>
 *   <li>Targets shape: [batch] (class indices: 0, 1, ..., C-1)</li>
 *   <li>Takes argmax of predictions</li>
 * </ul>
 *
 * <h3>Limitations</h3>
 * <ul>
 *   <li>Not suitable for imbalanced datasets (use Precision, Recall, F1Score)</li>
 *   <li>No confidence threshold tuning</li>
 *   <li>Treats all errors equally</li>
 * </ul>
 *
 * @see Precision
 * @see Recall
 * @see F1Score
 */
public class Accuracy {

    private long correct = 0;
    private long total = 0;

    /**
     * Update metric with a batch of predictions and targets.
     *
     * @param predictions predicted logits/probabilities, shape: [batch, num_classes] or [batch]
     * @param targets target class indices or binary labels, shape: [batch]
     *
     * @throws IllegalArgumentException if shapes are incompatible
     */
    public void update(float[] predictions, float[] targets) {
        if (predictions.length != targets.length) {
            throw new IllegalArgumentException(
                "predictions and targets must have same batch size, got: " +
                predictions.length + " vs " + targets.length
            );
        }

        for (int i = 0; i < targets.length; i++) {
            int predicted = (int) Math.round(predictions[i]);
            int target = (int) targets[i];
            if (predicted == target) {
                correct++;
            }
            total++;
        }
    }

    /**
     * Update metric with multi-class predictions and targets.
     *
     * @param predictions predicted logits, shape: [batch, num_classes]
     * @param targets target class indices, shape: [batch]
     * @param numClasses number of classes
     */
    public void updateMulticlass(float[][] predictions, int[] targets, int numClasses) {
        if (predictions.length != targets.length) {
            throw new IllegalArgumentException("batch size mismatch");
        }

        for (int i = 0; i < targets.length; i++) {
            int predictedClass = argmax(predictions[i]);
            int targetClass = targets[i];
            if (predictedClass == targetClass) {
                correct++;
            }
            total++;
        }
    }

    /**
     * Compute accuracy metric.
     *
     * @return accuracy as a fraction (0 to 1)
     */
    public float compute() {
        if (total == 0) {
            return 0f;
        }
        return (float) correct / total;
    }

    /**
     * Reset metric state.
     */
    public void reset() {
        correct = 0;
        total = 0;
    }

    /**
     * Get number of correct predictions.
     *
     * @return count of correct predictions
     */
    public long getCorrect() {
        return correct;
    }

    /**
     * Get total number of predictions.
     *
     * @return total predictions processed
     */
    public long getTotal() {
        return total;
    }

    private int argmax(float[] arr) {
        int maxIdx = 0;
        float maxVal = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    @Override
    public String toString() {
        return String.format("Accuracy(correct=%d, total=%d, acc=%.4f)", correct, total, compute());
    }
}
