package tech.kayys.gollek.ml.tree;

import tech.kayys.gollek.ml.base.BaseEstimator;
import java.util.*;

/**
 * Decision Tree Regressor using MSE for splitting.
 */
public class DecisionTreeRegressor extends BaseEstimator {

    private final int maxDepth;
    private final int minSamplesSplit;
    private Node root;
    private boolean isFitted = false;

    public DecisionTreeRegressor() {
        this(Integer.MAX_VALUE, 2);
    }

    public DecisionTreeRegressor(int maxDepth, int minSamplesSplit) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        double[] yd = new double[y.length];
        for (int i = 0; i < y.length; i++) yd[i] = y[i];
        fit(X, yd);
    }

    public void fit(float[][] X, double[] y) {
        validateData(X, new int[y.length]); // Dummy labels for validation
        root = buildTree(X, y, 0);
        setFitted(true);
        this.isFitted = true;
    }

    private Node buildTree(float[][] X, double[] y, int depth) {
        int nSamples = X.length;
        if (nSamples == 0) return null;
        int nFeatures = X[0].length;

        // Base cases
        if (depth >= maxDepth || nSamples < minSamplesSplit || allSame(y)) {
            return new Node(mean(y));
        }

        // Find best split
        int bestFeature = -1;
        float bestThreshold = 0;
        double minMse = Double.MAX_VALUE;

        for (int j = 0; j < nFeatures; j++) {
            float[] featureValues = getFeatureValues(X, j);
            float[] thresholds = getThresholds(featureValues);

            for (float threshold : thresholds) {
                SplitResult split = split(X, y, j, threshold);
                if (split.leftY.length == 0 || split.rightY.length == 0) continue;

                double mse = computeWeightedMse(split.leftY, split.rightY);
                if (mse < minMse) {
                    minMse = mse;
                    bestFeature = j;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature == -1) return new Node(mean(y));

        SplitResult bestSplit = split(X, y, bestFeature, bestThreshold);
        Node node = new Node(bestFeature, bestThreshold);
        node.left = buildTree(bestSplit.leftX, bestSplit.leftY, depth + 1);
        node.right = buildTree(bestSplit.rightX, bestSplit.rightY, depth + 1);
        return node;
    }

    private double computeWeightedMse(double[] leftY, double[] rightY) {
        int nL = leftY.length;
        int nR = rightY.length;
        int total = nL + nR;
        return (nL * mse(leftY) + nR * mse(rightY)) / total;
    }

    private double mse(double[] y) {
        double mean = mean(y);
        double sum = 0;
        for (double val : y) {
            double diff = val - mean;
            sum += diff * diff;
        }
        return sum / y.length;
    }

    private double mean(double[] y) {
        if (y.length == 0) return 0;
        double sum = 0;
        for (double val : y) sum += val;
        return sum / y.length;
    }

    private boolean allSame(double[] y) {
        for (int i = 1; i < y.length; i++) {
            if (y[i] != y[0]) return false;
        }
        return true;
    }

    private float[] getFeatureValues(float[][] X, int j) {
        float[] values = new float[X.length];
        for (int i = 0; i < X.length; i++) values[i] = X[i][j];
        return values;
    }

    private float[] getThresholds(float[] values) {
        float[] sorted = values.clone();
        Arrays.sort(sorted);
        List<Float> thresholds = new ArrayList<>();
        for (int i = 0; i < sorted.length - 1; i++) {
            if (sorted[i] != sorted[i+1]) {
                thresholds.add((sorted[i] + sorted[i+1]) / 2.0f);
            }
        }
        float[] result = new float[thresholds.size()];
        for (int i = 0; i < thresholds.size(); i++) result[i] = thresholds.get(i);
        return result;
    }

    private SplitResult split(float[][] X, double[] y, int feature, float threshold) {
        int nSamples = X.length;
        int leftCount = 0;
        for (int i = 0; i < nSamples; i++) {
            if (X[i][feature] <= threshold) leftCount++;
        }

        float[][] leftX = new float[leftCount][];
        double[] leftY = new double[leftCount];
        float[][] rightX = new float[nSamples - leftCount][];
        double[] rightY = new double[nSamples - leftCount];

        int l = 0, r = 0;
        for (int i = 0; i < nSamples; i++) {
            if (X[i][feature] <= threshold) {
                leftX[l] = X[i];
                leftY[l++] = y[i];
            } else {
                rightX[r] = X[i];
                rightY[r++] = y[i];
            }
        }
        return new SplitResult(leftX, leftY, rightX, rightY);
    }

    @Override
    public int[] predict(float[][] X) {
        validateInput(X);
        int[] preds = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            preds[i] = (int) Math.round(predictSingleValue(X[i]));
        }
        return preds;
    }

    @Override
    public double[] predictValues(float[][] X) {
        validateInput(X);
        double[] vals = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            vals[i] = predictSingleValue(X[i]);
        }
        return vals;
    }

    @Override
    public double predictSingleValue(float[] x) {
        Node current = root;
        if (current == null) return 0;
        while (!current.isLeaf()) {
            if (x[current.feature] <= current.threshold) {
                current = current.left;
            } else {
                current = current.right;
            }
        }
        return current.value;
    }

    @Override
    public boolean isFitted() {
        return isFitted;
    }

    public void setRoot(Node root) {
        this.root = root;
        this.setFitted(true);
        this.isFitted = true;
    }

    public static class Node {
        public int feature;
        public float threshold;
        public double value;
        public Node left, right;

        public Node(double value) {
            this.value = value;
            this.feature = -1;
        }

        public Node(int feature, float threshold) {
            this.feature = feature;
            this.threshold = threshold;
        }

        public boolean isLeaf() {
            return feature == -1;
        }
    }

    private static class SplitResult {
        float[][] leftX, rightX;
        double[] leftY, rightY;

        SplitResult(float[][] lx, double[] ly, float[][] rx, double[] ry) {
            leftX = lx; leftY = ly;
            rightX = rx; rightY = ry;
        }
    }
}
