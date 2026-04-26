package tech.kayys.gollek.ml.svm;

import tech.kayys.gollek.ml.base.BaseEstimator;
import java.util.*;
import java.util.stream.*;

/**
 * Support Vector Machine (SVM) with SMO algorithm.
 * Supports linear and RBF kernels.
 */
public class SVC extends BaseEstimator {

    private final double C;
    private final String kernel; // linear, rbf, poly
    private final double gammaValue;
    private final String gammaType;
    private final int degree;
    private final double coef0;
    private final double tol;
    private final int maxIter;

    private int nFeatures;
    private double[] alpha; // Lagrange multipliers
    private double b; // Bias
    private float[][] supportVectors;
    private int[] supportVectorLabels;
    private double[] supportVectorAlphas;
    private boolean isFitted = false;

    public SVC() {
        this(1.0, "rbf", "scale", 3, 0.0, 1e-3, 1000);
    }

    public SVC(double C, String kernel, String gamma, int degree,
            double coef0, double tol, int maxIter) {
        this.C = C;
        this.kernel = kernel;
        this.gammaType = gamma;
        this.gammaValue = gamma.equals("scale") || gamma.equals("auto") ? 0.0 : Double.parseDouble(gamma);
        this.degree = degree;
        this.coef0 = coef0;
        this.tol = tol;
        this.maxIter = maxIter;
    }

    private double getGamma(int nFeatures) {
        if (gammaType.equals("scale")) return 1.0 / (nFeatures * 1.0); // Simplified var(X)
        if (gammaType.equals("auto")) return 1.0 / nFeatures;
        return gammaValue;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        validateData(X, y);
        int nSamples = X.length;
        this.nFeatures = X[0].length;
        double currentGamma = getGamma(nFeatures);

        // Convert labels to +1/-1
        int[] yb = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            yb[i] = y[i] == 0 ? -1 : 1;
        }

        // Initialize alpha
        alpha = new double[nSamples];
        b = 0;

        // Pre-compute kernel matrix
        double[][] K = computeKernelMatrix(X, currentGamma);

        // SMO algorithm
        int iter = 0;
        int entireSet = 1;
        int nChanged = 0;

        while (iter < maxIter && (nChanged > 0 || entireSet > 0)) {
            nChanged = 0;

            if (entireSet > 0) {
                // Check all examples
                for (int i = 0; i < nSamples; i++) {
                    nChanged += examineExample(i, X, yb, K, currentGamma);
                }
            } else {
                // Check non-bound examples only
                for (int i = 0; i < nSamples; i++) {
                    if (alpha[i] > 0 && alpha[i] < C) {
                        nChanged += examineExample(i, X, yb, K, currentGamma);
                    }
                }
            }

            iter++;

            if (entireSet > 0) {
                entireSet = 0;
            } else if (nChanged == 0) {
                entireSet = 1;
            }
        }

        // Extract support vectors
        List<float[]> svList = new ArrayList<>();
        List<Integer> svLabelList = new ArrayList<>();
        List<Double> svAlphaList = new ArrayList<>();

        for (int i = 0; i < nSamples; i++) {
            if (alpha[i] > tol) {
                svList.add(X[i].clone());
                svLabelList.add(yb[i]);
                svAlphaList.add(alpha[i]);
            }
        }

        supportVectors = svList.toArray(new float[0][]);
        supportVectorLabels = svLabelList.stream().mapToInt(i -> i).toArray();
        supportVectorAlphas = svAlphaList.stream().mapToDouble(d -> d).toArray();
        this.isFitted = true;
    }

    private int examineExample(int i2, float[][] X, int[] y, double[][] K, double gamma) {
        double yi = y[i2];
        double alphai = alpha[i2];
        double Ei = computeError(i2, X, y, K);
        double ri = yi * Ei;

        if ((ri < -tol && alphai < C) || (ri > tol && alphai > 0)) {
            int i1 = -1;
            double maxStep = 0;

            for (int j = 0; j < X.length; j++) {
                if (j != i2 && alpha[j] > 0 && alpha[j] < C) {
                    double Ej = computeError(j, X, y, K);
                    double step = Math.abs(Ei - Ej);
                    if (step > maxStep) {
                        maxStep = step;
                        i1 = j;
                    }
                }
            }

            if (i1 != -1 && takeStep(i1, i2, X, y, K)) return 1;

            for (int j = 0; j < X.length; j++) {
                if (j != i2 && alpha[j] > 0 && alpha[j] < C) {
                    if (takeStep(j, i2, X, y, K)) return 1;
                }
            }

            for (int j = 0; j < X.length; j++) {
                if (j != i2) {
                    if (takeStep(j, i2, X, y, K)) return 1;
                }
            }
        }
        return 0;
    }

    private boolean takeStep(int i1, int i2, float[][] X, int[] y, double[][] K) {
        if (i1 == i2) return false;

        double alph1 = alpha[i1];
        double alph2 = alpha[i2];
        double y1 = y[i1];
        double y2 = y[i2];
        double E1 = computeError(i1, X, y, K);
        double E2 = computeError(i2, X, y, K);
        double s = y1 * y2;

        double L, H;
        if (y1 != y2) {
            L = Math.max(0, alph2 - alph1);
            H = Math.min(C, C + alph2 - alph1);
        } else {
            L = Math.max(0, alph1 + alph2 - C);
            H = Math.min(C, alph1 + alph2);
        }

        if (Math.abs(L - H) < tol) return false;

        double eta = 2 * K[i1][i2] - K[i1][i1] - K[i2][i2];

        double newAlph2;
        if (eta < 0) {
            newAlph2 = alph2 - y2 * (E1 - E2) / eta;
            if (newAlph2 < L) newAlph2 = L;
            else if (newAlph2 > H) newAlph2 = H;
        } else {
            return false;
        }

        if (Math.abs(newAlph2 - alph2) < tol * (newAlph2 + alph2 + tol)) return false;

        double newAlph1 = alph1 + s * (alph2 - newAlph2);

        double b1 = b - E1 - y1 * (newAlph1 - alph1) * K[i1][i1] - y2 * (newAlph2 - alph2) * K[i1][i2];
        double b2 = b - E2 - y1 * (newAlph1 - alph1) * K[i1][i2] - y2 * (newAlph2 - alph2) * K[i2][i2];

        b = (b1 + b2) / 2;
        alpha[i1] = newAlph1;
        alpha[i2] = newAlph2;

        return true;
    }

    private double computeError(int i, float[][] X, int[] y, double[][] K) {
        double sum = -b;
        for (int j = 0; j < X.length; j++) {
            if (alpha[j] > 0) sum += alpha[j] * y[j] * K[i][j];
        }
        return sum - y[i];
    }

    private double[][] computeKernelMatrix(float[][] X, double gamma) {
        int n = X.length;
        double[][] K = new double[n][n];
        IntStream.range(0, n).parallel().forEach(i -> {
            for (int j = 0; j < n; j++) {
                K[i][j] = kernelFunction(X[i], X[j], gamma);
            }
        });
        return K;
    }

    private double kernelFunction(float[] x1, float[] x2, double gamma) {
        switch (kernel) {
            case "linear": return dot(x1, x2);
            case "rbf":
                double sqDist = 0;
                for (int i = 0; i < x1.length; i++) {
                    double diff = x1[i] - x2[i];
                    sqDist += diff * diff;
                }
                return Math.exp(-gamma * sqDist);
            case "poly":
                return Math.pow(gamma * dot(x1, x2) + coef0, degree);
            default: return dot(x1, x2);
        }
    }

    private double dot(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    @Override
    public int[] predict(float[][] X) {
        validateInput(X);
        double currentGamma = getGamma(nFeatures);
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            double sum = -b;
            for (int j = 0; j < supportVectors.length; j++) {
                sum += supportVectorAlphas[j] * supportVectorLabels[j] * kernelFunction(X[i], supportVectors[j], currentGamma);
            }
            predictions[i] = sum > 0 ? 1 : 0;
        }
        return predictions;
    }

    public void setSupportVectors(float[][] supportVectors, int[] labels, double[] alphas, double b) {
        this.supportVectors = supportVectors;
        this.supportVectorLabels = labels;
        this.supportVectorAlphas = alphas;
        this.b = b;
        this.isFitted = true;
        this.nFeatures = supportVectors.length > 0 ? supportVectors[0].length : 0;
    }

    @Override
    public boolean isFitted() {
        return isFitted;
    }
}