package tech.kayys.gollek.ml.naive_bayes;

import tech.kayys.gollek.ml.base.BaseEstimator;
import java.util.*;

/**
 * Multinomial Naive Bayes for discrete features (e.g., text counts).
 */
public class MultinomialNB extends BaseEstimator {
    private double[][] classLogProbs;
    private double[] classPriors;
    private double alpha = 1.0; // Laplace smoothing
    private int nClasses;
    private boolean isFitted = false;

    public MultinomialNB() {}

    public MultinomialNB(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        validateData(X, y);
        nClasses = Arrays.stream(y).max().getAsInt() + 1;
        int nFeatures = X[0].length;

        classLogProbs = new double[nClasses][nFeatures];
        classPriors = new double[nClasses];

        double[] featureSumPerClass = new double[nClasses];
        double[][] featureCounts = new double[nClasses][nFeatures];

        // Count features per class
        for (int i = 0; i < X.length; i++) {
            int c = y[i];
            classPriors[c]++;
            for (int j = 0; j < nFeatures; j++) {
                featureCounts[c][j] += X[i][j];
                featureSumPerClass[c] += X[i][j];
            }
        }

        // Normalize to probabilities with Laplace smoothing
        for (int c = 0; c < nClasses; c++) {
            classPriors[c] /= X.length;
            double denominator = featureSumPerClass[c] + alpha * nFeatures;

            for (int j = 0; j < nFeatures; j++) {
                double prob = (featureCounts[c][j] + alpha) / denominator;
                classLogProbs[c][j] = Math.log(prob);
            }
        }
        this.isFitted = true;
    }

    @Override
    public int[] predict(float[][] X) {
        validateInput(X);
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predictSingleSample(X[i]);
        }
        return predictions;
    }

    private int predictSingleSample(float[] x) {
        double[] logProbs = new double[nClasses];
        for (int c = 0; c < nClasses; c++) {
            logProbs[c] = Math.log(classPriors[c] + 1e-11);
            for (int j = 0; j < x.length; j++) {
                if (x[j] > 0) {
                    logProbs[c] += x[j] * classLogProbs[c][j];
                }
            }
        }
        int maxIdx = 0;
        for (int i = 1; i < nClasses; i++) {
            if (logProbs[i] > logProbs[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    @Override
    public boolean isFitted() {
        return isFitted;
    }
}
