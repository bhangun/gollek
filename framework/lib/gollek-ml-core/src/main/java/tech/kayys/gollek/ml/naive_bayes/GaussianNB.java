package tech.kayys.gollek.ml.naive_bayes;

import tech.kayys.gollek.ml.base.BaseEstimator;
import java.util.*;

/**
 * Gaussian Naive Bayes for continuous features.
 */
public class GaussianNB extends BaseEstimator {
    private double[][] means;
    private double[][] variances;
    private double[] classPriors;
    private int nClasses;
    private double varSmoothing = 1e-9;
    private boolean isFitted = false;

    @Override
    public void fit(float[][] X, int[] y) {
        validateData(X, y);
        nClasses = Arrays.stream(y).max().getAsInt() + 1;
        int nFeatures = X[0].length;

        means = new double[nClasses][nFeatures];
        variances = new double[nClasses][nFeatures];
        classPriors = new double[nClasses];

        // Collect samples per class
        List<List<float[]>> classSamples = new ArrayList<>(nClasses);
        for (int c = 0; c < nClasses; c++) {
            classSamples.add(new ArrayList<>());
        }

        for (int i = 0; i < X.length; i++) {
            classSamples.get(y[i]).add(X[i]);
        }

        // Compute statistics for each class
        for (int c = 0; c < nClasses; c++) {
            List<float[]> samples = classSamples.get(c);
            classPriors[c] = (double) samples.size() / X.length;

            if (samples.isEmpty()) continue;

            // Compute means
            for (int j = 0; j < nFeatures; j++) {
                double sum = 0;
                for (float[] sample : samples) {
                    sum += sample[j];
                }
                means[c][j] = sum / samples.size();
            }

            // Compute variances
            for (int j = 0; j < nFeatures; j++) {
                double sumSq = 0;
                for (float[] sample : samples) {
                    double diff = sample[j] - means[c][j];
                    sumSq += diff * diff;
                }
                variances[c][j] = sumSq / samples.size() + varSmoothing;
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
        double[] logProbs = predictLogProbaSingle(x);
        int maxIdx = 0;
        for (int i = 1; i < logProbs.length; i++) {
            if (logProbs[i] > logProbs[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    @Override
    public double[][] predictProba(float[][] X) {
        validateInput(X);
        double[][] probabilities = new double[X.length][nClasses];
        for (int i = 0; i < X.length; i++) {
            probabilities[i] = predictProbaSingle(X[i]);
        }
        return probabilities;
    }

    public double[] predictLogProbaSingle(float[] x) {
        double[] logProbs = new double[nClasses];
        for (int c = 0; c < nClasses; c++) {
            logProbs[c] = Math.log(classPriors[c] + 1e-11);
            for (int j = 0; j < x.length; j++) {
                double diff = x[j] - means[c][j];
                logProbs[c] -= 0.5 * Math.log(2 * Math.PI * variances[c][j]);
                logProbs[c] -= 0.5 * diff * diff / variances[c][j];
            }
        }
        return logProbs;
    }

    public double[] predictProbaSingle(float[] x) {
        double[] logProbs = predictLogProbaSingle(x);
        double maxLog = Arrays.stream(logProbs).max().getAsDouble();
        double[] probs = new double[nClasses];
        double sum = 0;
        for (int c = 0; c < nClasses; c++) {
            probs[c] = Math.exp(logProbs[c] - maxLog);
            sum += probs[c];
        }
        for (int c = 0; c < nClasses; c++) probs[c] /= sum;
        return probs;
    }

    @Override
    public boolean isFitted() {
        return isFitted;
    }
}