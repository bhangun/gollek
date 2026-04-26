package tech.kayys.gollek.ml.tree;

import tech.kayys.gollek.ml.base.BaseEstimator;
import java.util.*;
import java.util.stream.IntStream;

/**
 * Decision Tree with Gini/Entropy splitting criteria.
 * Parallel tree building with SIMD optimizations.
 */
public class DecisionTreeClassifier extends BaseEstimator {

    private Node root;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int minSamplesLeaf;
    private final String criterion; // "gini" or "entropy"
    private final double maxFeatures;
    private final Random random;

    private int nClasses;
    private int nFeatures;
    private double[] featureImportances;

    public DecisionTreeClassifier() {
        this(5, 2, 1, "gini", "sqrt");
    }

    public DecisionTreeClassifier(int maxDepth, int minSamplesSplit,
            int minSamplesLeaf, String criterion,
            String maxFeatures) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.criterion = criterion;
        this.maxFeatures = "sqrt".equals(maxFeatures) ? -1.0 : "log2".equals(maxFeatures) ? -2.0 : Double.parseDouble(maxFeatures);
        this.random = new Random();
    }

    @Override
    public void fit(float[][] X, int[] y) {
        validateData(X, y);
        this.nFeatures = X[0].length;
        this.nClasses = (int) Arrays.stream(y).distinct().count();

        // Resolve maxFeatures
        double mf;
        if (maxFeatures == -1.0) mf = Math.sqrt(nFeatures);
        else if (maxFeatures == -2.0) mf = Math.log(nFeatures) / Math.log(2);
        else if (maxFeatures <= 1.0) mf = maxFeatures * nFeatures;
        else mf = maxFeatures;
        int nFeaturesToConsider = (int) Math.max(1, Math.min(nFeatures, mf));

        // Build tree recursively
        int[] indices = IntStream.range(0, X.length).toArray();
        this.root = buildTree(X, y, 0, indices, nFeaturesToConsider);

        // Calculate feature importances
        featureImportances = new double[nFeatures];
        computeImportances(root, featureImportances);

        // Normalize importances
        double total = Arrays.stream(featureImportances).sum();
        if (total > 0) {
            for (int i = 0; i < featureImportances.length; i++) {
                featureImportances[i] /= total;
            }
        }
    }

    private Node buildTree(float[][] X, int[] y, int depth, int[] indices, int nFeaturesToConsider) {
        if (depth >= maxDepth || indices.length < minSamplesSplit || isPure(y, indices)) {
            return new LeafNode(getMajorityClass(y, indices), getClassProbs(y, indices));
        }

        Split bestSplit = findBestSplitParallel(X, y, indices, nFeaturesToConsider);

        if (bestSplit == null || bestSplit.gain < 1e-8) {
            return new LeafNode(getMajorityClass(y, indices), getClassProbs(y, indices));
        }

        int[] leftIndices = new int[bestSplit.leftCount];
        int[] rightIndices = new int[bestSplit.rightCount];
        int leftIdx = 0, rightIdx = 0;

        for (int idx : indices) {
            if (X[idx][bestSplit.feature] <= bestSplit.threshold) {
                leftIndices[leftIdx++] = idx;
            } else {
                rightIndices[rightIdx++] = idx;
            }
        }

        Node left = buildTree(X, y, depth + 1, leftIndices, nFeaturesToConsider);
        Node right = buildTree(X, y, depth + 1, rightIndices, nFeaturesToConsider);

        return new SplitNode(bestSplit.feature, bestSplit.threshold, left, right, bestSplit.gain);
    }

    private Split findBestSplitParallel(float[][] X, int[] y, int[] indices, int nFeaturesToConsider) {
        int[] featureIndices = selectRandomFeatures(nFeaturesToConsider);

        return Arrays.stream(featureIndices)
                .parallel()
                .mapToObj(feature -> evaluateFeature(X, y, indices, feature))
                .filter(Objects::nonNull)
                .max((a, b) -> Double.compare(a.gain, b.gain))
                .orElse(null);
    }

    private Split evaluateFeature(float[][] X, int[] y, int[] indices, int feature) {
        double[] values = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            values[i] = X[indices[i]][feature];
        }

        Integer[] sortedIndices = new Integer[indices.length];
        for (int i = 0; i < indices.length; i++) sortedIndices[i] = i;
        Arrays.sort(sortedIndices, Comparator.comparingDouble(a -> values[a]));

        Split bestSplit = null;
        double bestGain = 0;

        int[] leftCounts = new int[nClasses];
        int[] rightCounts = new int[nClasses];
        for (int idx : indices) rightCounts[y[idx]]++;

        int totalLeft = 0;
        int totalSamples = indices.length;

        for (int i = 0; i < totalSamples - 1; i++) {
            int idx = sortedIndices[i];
            int currentClass = y[indices[idx]];

            rightCounts[currentClass]--;
            leftCounts[currentClass]++;
            totalLeft++;

            if (values[idx] == values[sortedIndices[i + 1]]) continue;
            if (totalLeft < minSamplesLeaf || (totalSamples - totalLeft) < minSamplesLeaf) continue;

            double currentImpurity = calculateImpurity(leftCounts, rightCounts, totalLeft, totalSamples - totalLeft);
            double gain = calculateImpurity(getClassCounts(y, indices)) - currentImpurity;

            if (gain > bestGain) {
                bestGain = gain;
                bestSplit = new Split(feature, (values[idx] + values[sortedIndices[i+1]]) / 2.0, totalLeft, totalSamples - totalLeft, bestGain);
            }
        }
        return bestSplit;
    }

    private int[] getClassCounts(int[] y, int[] indices) {
        int[] counts = new int[nClasses];
        for (int idx : indices) counts[y[idx]]++;
        return counts;
    }

    private double calculateImpurity(int[] counts) {
        int total = Arrays.stream(counts).sum();
        if (total == 0) return 0;
        double impurity = ("gini".equals(criterion)) ? 1.0 : 0.0;
        for (int count : counts) {
            double p = count / (double) total;
            if (p > 0) {
                if ("gini".equals(criterion)) impurity -= p * p;
                else impurity -= p * Math.log(p) / Math.log(2);
            }
        }
        return impurity;
    }

    private double calculateImpurity(int[] leftCounts, int[] rightCounts, int leftTotal, int rightTotal) {
        int total = leftTotal + rightTotal;
        return (leftTotal / (double) total) * calculateImpurity(leftCounts) +
               (rightTotal / (double) total) * calculateImpurity(rightCounts);
    }

    private boolean isPure(int[] y, int[] indices) {
        int first = y[indices[0]];
        for (int i = 1; i < indices.length; i++) {
            if (y[indices[i]] != first) return false;
        }
        return true;
    }

    private int getMajorityClass(int[] y, int[] indices) {
        int[] counts = getClassCounts(y, indices);
        int maxIdx = 0;
        for (int i = 1; i < nClasses; i++) {
            if (counts[i] > counts[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    private double[] getClassProbs(int[] y, int[] indices) {
        double[] probs = new double[nClasses];
        for (int idx : indices) probs[y[idx]]++;
        for (int i = 0; i < nClasses; i++) probs[i] /= indices.length;
        return probs;
    }

    private int[] selectRandomFeatures(int k) {
        int[] features = IntStream.range(0, nFeatures).toArray();
        for (int i = 0; i < k; i++) {
            int j = i + random.nextInt(nFeatures - i);
            int temp = features[i];
            features[i] = features[j];
            features[j] = temp;
        }
        return Arrays.copyOf(features, k);
    }

    private void computeImportances(Node node, double[] importances) {
        if (node instanceof SplitNode) {
            SplitNode split = (SplitNode) node;
            importances[split.feature] += split.gain;
            computeImportances(split.left, importances);
            computeImportances(split.right, importances);
        }
    }

    @Override
    public boolean isFitted() {
        return root != null;
    }

    @Override
    public int[] predict(float[][] X) {
        validateInput(X);
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predictSingle(X[i]);
        }
        return predictions;
    }

    @Override
    public double[][] predictProba(float[][] X) {
        validateInput(X);
        double[][] probs = new double[X.length][nClasses];
        for (int i = 0; i < X.length; i++) {
            probs[i] = predictProbaSingle(X[i]);
        }
        return probs;
    }

    public int predictSingle(float[] x) {
        Node node = root;
        while (node instanceof SplitNode) {
            SplitNode split = (SplitNode) node;
            node = x[split.feature] <= split.threshold ? split.left : split.right;
        }
        return ((LeafNode) node).prediction;
    }

    public double[] predictProbaSingle(float[] x) {
        Node node = root;
        while (node instanceof SplitNode) {
            SplitNode split = (SplitNode) node;
            node = x[split.feature] <= split.threshold ? split.left : split.right;
        }
        return ((LeafNode) node).probabilities;
    }

    public double[] featureImportances() {
        return featureImportances;
    }

    public void setRoot(Node root) {
        this.root = root;
        this.setFitted(true);
    }

    public static abstract class Node {}

    public static class SplitNode extends Node {
        public final int feature;
        public final double threshold;
        public final Node left, right;
        public final double gain;
        public SplitNode(int feature, double threshold, Node left, Node right, double gain) {
            this.feature = feature; this.threshold = threshold; this.left = left; this.right = right; this.gain = gain;
        }
    }

    public static class LeafNode extends Node {
        public final int prediction;
        public final double[] probabilities;
        public LeafNode(int prediction, double[] probabilities) {
            this.prediction = prediction;
            this.probabilities = probabilities;
        }
    }

    private static class Split {
        final int feature;
        final double threshold;
        final int leftCount;
        final int rightCount;
        final double gain;
        Split(int feature, double threshold, int leftCount, int rightCount, double gain) {
            this.feature = feature; this.threshold = threshold; this.leftCount = leftCount; this.rightCount = rightCount; this.gain = gain;
        }
    }
}