package tech.kayys.gollek.ml.linear_model;

import java.util.*;
import java.util.concurrent.*;
import tech.kayys.gollek.ml.base.BaseEstimator;

/**
 * Linear regression with various regularizations.
 */
public class LinearModel extends BaseEstimator {
    private final String penalty; // "l1", "l2", "elasticnet", "none"
    private final double alpha;
    private final double l1Ratio;
    private final double tol;
    private final int maxIter;
    private final double learningRate;
    private double[] coefficients;
    private double intercept;

    public LinearModel() {
        this("l2", 1.0, 0.5, 1e-4, 1000, 0.01);
    }

    public LinearModel(String penalty, double alpha, double l1Ratio,
            double tol, int maxIter, double learningRate) {
        this.penalty = penalty;
        this.alpha = alpha;
        this.l1Ratio = l1Ratio;
        this.tol = tol;
        this.maxIter = maxIter;
        this.learningRate = learningRate;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        double[] yd = new double[y.length];
        for (int i = 0; i < y.length; i++) yd[i] = y[i];
        fit(X, yd);
    }

    public void fit(float[][] X, float[] y) {
        double[] yd = new double[y.length];
        for (int i = 0; i < y.length; i++) yd[i] = y[i];
        fit(X, yd);
    }

    public void fit(float[][] X, double[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        double[][] Xd = new double[nSamples][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                Xd[i][j] = X[i][j];
            }
        }
        fitDouble(Xd, y);
        setFitted(true);
    }

    private void fitDouble(double[][] X, double[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;

        // Standardize features
        double[] means = new double[nFeatures];
        double[] stds = new double[nFeatures];
        double[][] Xs = new double[nSamples][nFeatures];

        for (int j = 0; j < nFeatures; j++) {
            double sum = 0;
            for (int i = 0; i < nSamples; i++) {
                sum += X[i][j];
            }
            means[j] = sum / nSamples;

            double var = 0;
            for (int i = 0; i < nSamples; i++) {
                double diff = X[i][j] - means[j];
                var += diff * diff;
            }
            stds[j] = Math.sqrt(var / (nSamples - 1));
            if (stds[j] < 1e-8) stds[j] = 1;

            for (int i = 0; i < nSamples; i++) {
                Xs[i][j] = (X[i][j] - means[j]) / stds[j];
            }
        }

        // Center target
        double yMean = 0;
        for (double v : y) yMean += v;
        yMean /= nSamples;
        double[] ys = new double[nSamples];
        for (int i = 0; i < nSamples; i++) ys[i] = y[i] - yMean;

        // Initialize coefficients
        coefficients = new double[nFeatures];

        if ("none".equals(penalty)) {
            fitOLS(Xs, ys);
        } else {
            fitCoordinateDescent(Xs, ys);
        }

        // Adjust intercept
        intercept = yMean;
        for (int j = 0; j < nFeatures; j++) {
            intercept -= coefficients[j] * means[j] / stds[j];
        }

        // Transform coefficients back to original scale
        for (int j = 0; j < nFeatures; j++) {
            coefficients[j] /= stds[j];
        }
    }

    private void fitOLS(double[][] X, double[] y) {
        int n = X.length;
        int p = X[0].length;
        double[][] XTX = new double[p][p];
        double[] XTy = new double[p];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                XTy[j] += X[i][j] * y[i];
                for (int k = 0; k <= j; k++) {
                    XTX[j][k] += X[i][j] * X[i][k];
                }
            }
        }

        for (int j = 0; j < p; j++) {
            for (int k = j + 1; k < p; k++) {
                XTX[j][k] = XTX[k][j];
            }
        }

        coefficients = solveCholesky(XTX, XTy);
    }

    private void fitCoordinateDescent(double[][] X, double[] y) {
        int n = X.length;
        int p = coefficients.length;
        double[] residuals = new double[n];
        for (int i = 0; i < n; i++) {
            double pred = 0;
            for (int j = 0; j < p; j++) {
                pred += X[i][j] * coefficients[j];
            }
            residuals[i] = y[i] - pred;
        }

        for (int iter = 0; iter < maxIter; iter++) {
            double maxChange = 0;
            for (int j = 0; j < p; j++) {
                double xjRes = 0;
                for (int i = 0; i < n; i++) xjRes += X[i][j] * residuals[i];

                double oldCoef = coefficients[j];
                double newCoef = softThreshold(xjRes + oldCoef, alpha * l1Ratio) /
                        (1 + alpha * (1 - l1Ratio));

                if (newCoef != oldCoef) {
                    double delta = newCoef - oldCoef;
                    for (int i = 0; i < n; i++) residuals[i] -= delta * X[i][j];
                    coefficients[j] = newCoef;
                    maxChange = Math.max(maxChange, Math.abs(delta));
                }
            }
            if (maxChange < tol) break;
        }
    }

    private double softThreshold(double x, double lambda) {
        if (x > lambda) return x - lambda;
        if (x < -lambda) return x + lambda;
        return 0;
    }

    private double[] solveCholesky(double[][] A, double[] b) {
        int n = A.length;
        double[][] L = new double[n][n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int k = 0; k < i; k++) sum += L[i][k] * L[i][k];
            L[i][i] = Math.sqrt(A[i][i] - sum);
            for (int j = i + 1; j < n; j++) {
                sum = 0;
                for (int k = 0; k < i; k++) sum += L[j][k] * L[i][k];
                L[j][i] = (A[j][i] - sum) / L[i][i];
            }
        }
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < i; j++) sum += L[i][j] * y[j];
            y[i] = (b[i] - sum) / L[i][i];
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0;
            for (int j = i + 1; j < n; j++) sum += L[j][i] * x[j];
            x[i] = (y[i] - sum) / L[i][i];
        }
        return x;
    }

    @Override
    public int[] predict(float[][] X) {
        double[] preds = predictDouble(X);
        int[] predictions = new int[preds.length];
        for (int i = 0; i < preds.length; i++) {
            predictions[i] = (int) Math.round(preds[i]);
        }
        return predictions;
    }

    public double[] predictDouble(float[][] X) {
        validateInput(X);
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = intercept;
            for (int j = 0; j < X[i].length && j < coefficients.length; j++) {
                predictions[i] += X[i][j] * coefficients[j];
            }
        }
        return predictions;
    }

    public double[][] regularizationPath(float[][] X, double[] y, int nAlphas) {
        double[] alphas = new double[nAlphas];
        double[][] coefPath = new double[nAlphas][X[0].length];
        double alphaMax = 0;
        for (int j = 0; j < X[0].length; j++) {
            double corr = 0;
            for (int i = 0; i < X.length; i++) corr += X[i][j] * y[i];
            alphaMax = Math.max(alphaMax, Math.abs(corr));
        }
        alphaMax /= X.length;
        for (int i = 0; i < nAlphas; i++) {
            alphas[i] = alphaMax * Math.pow(0.1, (double) i / (nAlphas - 1));
        }
        for (int i = 0; i < nAlphas; i++) {
            LinearModel model = new LinearModel(penalty, alphas[i], l1Ratio, tol, maxIter, learningRate);
            model.fit(X, y);
            coefPath[i] = model.coefficients.clone();
        }
        return coefPath;
    }

    public void setCoefficients(double[] coefficients) {
        this.coefficients = coefficients;
        this.setFitted(true);
    }

    public void setIntercept(double intercept) {
        this.intercept = intercept;
    }

    public double[] getCoefficients() {
        return coefficients;
    }

    public double getIntercept() {
        return intercept;
    }
}