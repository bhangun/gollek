package tech.kayys.gollek.ml.ensemble;

import tech.kayys.gollek.ml.base.BaseEstimator;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Soft/Hard voting ensemble that combines multiple estimators.
 */
public class VotingClassifier extends BaseEstimator {
    private final List<BaseEstimator> estimators;
    private final String voting; // "hard" or "soft"
    private final double[] weights; // Weight for each estimator
    private int nClasses;
    private boolean isFitted = false;

    public VotingClassifier(List<BaseEstimator> estimators, String voting, double[] weights) {
        this.estimators = estimators;
        this.voting = voting;
        this.weights = weights != null ? weights : IntStream.range(0, estimators.size()).mapToDouble(i -> 1.0).toArray();
    }

    @Override
    public void fit(float[][] X, int[] y) {
        validateData(X, y);
        estimators.parallelStream().forEach(estimator -> estimator.fit(X, y));
        this.nClasses = (int) IntStream.of(y).distinct().count();
        this.isFitted = true;
    }

    @Override
    public int[] predict(float[][] X) {
        validateInput(X);
        int[] predictions = new int[X.length];

        if ("hard".equals(voting)) {
            // Hard voting: majority rule
            for (int i = 0; i < X.length; i++) {
                double[] votes = new double[nClasses];
                for (int e = 0; e < estimators.size(); e++) {
                    int pred = estimators.get(e).predictSingle(X[i]);
                    votes[pred] += weights[e];
                }
                predictions[i] = argmax(votes);
            }
        } else {
            // Soft voting: average probabilities
            double[][] finalProbs = new double[X.length][nClasses];

            for (int e = 0; e < estimators.size(); e++) {
                double[][] probs = estimators.get(e).predictProba(X);
                double w = weights[e];
                for (int i = 0; i < X.length; i++) {
                    for (int c = 0; c < nClasses; c++) {
                        finalProbs[i][c] += probs[i][c] * w;
                    }
                }
            }

            for (int i = 0; i < X.length; i++) {
                predictions[i] = argmax(finalProbs[i]);
            }
        }

        return predictions;
    }

    @Override
    public boolean isFitted() {
        return isFitted;
    }

    private int argmax(double[] values) {
        int bestIdx = 0;
        double maxVal = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > maxVal) {
                maxVal = values[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
}