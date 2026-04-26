package tech.kayys.gollek.ml.outlier;

import tech.kayys.gollek.ml.base.*;
import java.util.*;

/**
 * Isolation Forest for outlier detection.
 */
public class IsolationForest extends BaseEstimator {
    private final int nEstimators;
    private final int maxSamples;
    private final int maxFeatures;
    private final int randomState;
    private List<IsolationTree> trees;
    private double offset;

    public IsolationForest() {
        this(100, 256, 1, 42);
    }

    public IsolationForest(int nEstimators, int maxSamples, int maxFeatures, int randomState) {
        this.nEstimators = nEstimators;
        this.maxSamples = maxSamples;
        this.maxFeatures = maxFeatures;
        this.randomState = randomState;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        validateData(X, y);
        int nSamples = Math.min(maxSamples, X.length);
        int nFeatures = maxFeatures <= 0 ? X[0].length : Math.min(maxFeatures, X[0].length);

        trees = new ArrayList<>();
        Random rng = new Random(randomState);

        for (int i = 0; i < nEstimators; i++) {
            // Sample data
            List<float[]> sample = new ArrayList<>();
            for (int j = 0; j < nSamples; j++) {
                sample.add(X[rng.nextInt(X.length)]);
            }

            // Sample features
            List<Integer> featureIndices = new ArrayList<>();
            for (int j = 0; j < nFeatures; j++) {
                featureIndices.add(rng.nextInt(X[0].length));
            }

            IsolationTree tree = new IsolationTree();
            tree.fit(sample, featureIndices, rng);
            trees.add(tree);
        }

        // Calculate offset for anomaly scoring
        double[] scores = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            scores[i] = anomalyScore(X[i]);
        }

        // Use median as threshold
        Arrays.sort(scores);
        offset = scores[(int) (scores.length * 0.95)]; // 95th percentile
    }

    @Override
    public int[] predict(float[][] X) {
        validateInput(X);
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            double score = anomalyScore(X[i]);
            predictions[i] = score > offset ? -1 : 1;
        }
        return predictions;
    }

    public double[] scoreSamples(float[][] X) {
        validateInput(X);
        double[] scores = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            scores[i] = anomalyScore(X[i]);
        }
        return scores;
    }

    private double anomalyScore(float[] x) {
        double avgDepth = 0;
        for (IsolationTree tree : trees) {
            avgDepth += tree.pathLength(x);
        }
        avgDepth /= trees.size();

        // Expected path length for random tree
        double expected = expectedPathLength(maxSamples);
        return Math.pow(2, -avgDepth / expected);
    }

    static double expectedPathLength(int n) {
        if (n <= 1)
            return 0;
        if (n == 2)
            return 1;
        return 2 * (Math.log(n - 1) + 0.5772156649) - 2 * (n - 1.0) / n;
    }

    static class IsolationTree {
        private Node root;

        void fit(List<float[]> data, List<Integer> features, Random rng) {
            root = buildTree(data, features, 0, rng);
        }

        private Node buildTree(List<float[]> data, List<Integer> features, int depth, Random rng) {
            if (depth >= 100 || data.size() <= 1 || features.isEmpty()) {
                return new LeafNode(data.size());
            }

            // Randomly select feature and split value
            int featureIdx = features.get(rng.nextInt(features.size()));
            float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
            for (float[] point : data) {
                min = Math.min(min, point[featureIdx]);
                max = Math.max(max, point[featureIdx]);
            }

            if (min == max) {
                return new LeafNode(data.size());
            }

            float split = min + rng.nextFloat() * (max - min);

            // Split data
            List<float[]> left = new ArrayList<>();
            List<float[]> right = new ArrayList<>();
            for (float[] point : data) {
                if (point[featureIdx] < split) {
                    left.add(point);
                } else {
                    right.add(point);
                }
            }

            if (left.isEmpty() || right.isEmpty()) {
                return new LeafNode(data.size());
            }

            Node leftChild = buildTree(left, features, depth + 1, rng);
            Node rightChild = buildTree(right, features, depth + 1, rng);

            return new SplitNode(featureIdx, split, leftChild, rightChild);
        }

        double pathLength(float[] point) {
            return pathLength(root, point, 0);
        }

        private double pathLength(Node node, float[] point, int depth) {
            if (node instanceof LeafNode) {
                return depth + expectedPathLength(((LeafNode) node).size);
            }

            SplitNode split = (SplitNode) node;
            if (point[split.feature] < split.threshold) {
                return pathLength(split.left, point, depth + 1);
            } else {
                return pathLength(split.right, point, depth + 1);
            }
        }

        interface Node {
        }

        static class SplitNode implements Node {
            int feature;
            float threshold;
            Node left, right;

            SplitNode(int feature, float threshold, Node left, Node right) {
                this.feature = feature;
                this.threshold = threshold;
                this.left = left;
                this.right = right;
            }
        }

        static class LeafNode implements Node {
            int size;

            LeafNode(int size) {
                this.size = size;
            }
        }
    }

    @Override
    public boolean isFitted() {
        return trees != null && !trees.isEmpty();
    }
}