package tech.kayys.gollek.ml.nn.metrics;

import java.util.Arrays;

/**
 * Classification evaluation metrics: precision, recall, F1, accuracy,
 * confusion matrix, and top-K accuracy.
 *
 * <p>All multi-class metrics use <em>macro averaging</em> — each class is
 * weighted equally regardless of support.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * int[] predicted = {0, 1, 2, 1};
 * int[] actual    = {0, 1, 1, 2};
 * ClassificationMetrics.Result r =
 *     ClassificationMetrics.compute(predicted, actual, 3);
 * System.out.printf("F1=%.3f Acc=%.3f%n", r.f1(), r.accuracy());
 * }</pre>
 */
public final class ClassificationMetrics {

    private ClassificationMetrics() {}

    /**
     * Immutable result of a classification evaluation.
     *
     * @param precision macro-averaged precision across all classes
     * @param recall    macro-averaged recall across all classes
     * @param f1        macro-averaged F1 score across all classes
     * @param accuracy  overall accuracy (correct / total)
     */
    public record Result(float precision, float recall, float f1, float accuracy) {}

    /**
     * Computes macro-averaged precision, recall, F1, and accuracy.
     *
     * @param predicted   predicted class indices (length N)
     * @param actual      ground-truth class indices (length N)
     * @param numClasses  total number of classes
     * @return {@link Result} with all four metrics
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static Result compute(int[] predicted, int[] actual, int numClasses) {
        if (predicted.length != actual.length)
            throw new IllegalArgumentException("predicted and actual must have the same length");

        float[][] cm = confusionMatrix(predicted, actual, numClasses);
        float totalP = 0, totalR = 0, correct = 0;

        for (int c = 0; c < numClasses; c++) {
            float tp = cm[c][c];
            float fp = 0, fn = 0;
            for (int j = 0; j < numClasses; j++) {
                if (j != c) { fp += cm[j][c]; fn += cm[c][j]; }
            }
            totalP += (tp + fp) > 0 ? tp / (tp + fp) : 0f;
            totalR += (tp + fn) > 0 ? tp / (tp + fn) : 0f;
            correct += tp;
        }

        float p   = totalP / numClasses;
        float r   = totalR / numClasses;
        float f1  = (p + r) > 0 ? 2 * p * r / (p + r) : 0f;
        float acc = correct / predicted.length;
        return new Result(p, r, f1, acc);
    }

    /**
     * Builds a confusion matrix of shape {@code [numClasses][numClasses]}.
     *
     * <p>{@code matrix[actual][predicted]} counts how many times the true class
     * {@code actual} was predicted as {@code predicted}.
     *
     * @param predicted  predicted class indices
     * @param actual     ground-truth class indices
     * @param numClasses total number of classes
     * @return confusion matrix as {@code float[numClasses][numClasses]}
     */
    public static float[][] confusionMatrix(int[] predicted, int[] actual, int numClasses) {
        float[][] cm = new float[numClasses][numClasses];
        for (int i = 0; i < actual.length; i++) cm[actual[i]][predicted[i]]++;
        return cm;
    }

    /**
     * Computes top-K accuracy: the fraction of samples where the true label
     * appears in the K highest-scoring predictions.
     *
     * @param logits raw model outputs {@code [N, numClasses]}
     * @param labels ground-truth class indices (length N)
     * @param k      number of top predictions to consider
     * @return top-K accuracy in {@code [0, 1]}
     */
    public static float topKAccuracy(float[][] logits, int[] labels, int k) {
        int correct = 0;
        for (int i = 0; i < logits.length; i++) {
            int[] topK = topKIndices(logits[i], k);
            for (int idx : topK) if (idx == labels[i]) { correct++; break; }
        }
        return (float) correct / logits.length;
    }

    /** Returns the indices of the top-K values in descending order. */
    private static int[] topKIndices(float[] scores, int k) {
        Integer[] idx = new Integer[scores.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(scores[b], scores[a]));
        int[] result = new int[Math.min(k, idx.length)];
        for (int i = 0; i < result.length; i++) result[i] = idx[i];
        return result;
    }
}
