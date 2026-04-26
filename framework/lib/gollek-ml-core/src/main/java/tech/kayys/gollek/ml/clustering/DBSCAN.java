package tech.kayys.gollek.ml.clustering;

import tech.kayys.gollek.ml.base.BaseEstimator;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.IntStream;

/**
 * DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
 * Excellent for arbitrary-shaped clusters and outlier detection.
 */
public class DBSCAN extends BaseEstimator {
    private final double eps;
    private final int minSamples;
    private final String metric; // euclidean, manhattan, cosine
    private final int nJobs;

    private int[] labels;
    private int nClusters;
    private List<Integer> coreSampleIndices;

    public DBSCAN() {
        this(0.5, 5, "euclidean", -1);
    }

    public DBSCAN(double eps, int minSamples, String metric, int nJobs) {
        this.eps = eps;
        this.minSamples = minSamples;
        this.metric = metric;
        this.nJobs = nJobs > 0 ? nJobs : Runtime.getRuntime().availableProcessors();
    }

    @Override
    public void fit(float[][] X, int[] y) {
        fit(X);
    }

    public void fit(float[][] X) {
        int nSamples = X.length;
        labels = new int[nSamples];
        Arrays.fill(labels, -1); // -1 = noise

        int clusterId = 0;
        coreSampleIndices = new ArrayList<>();

        // Precompute neighbor lists in parallel using KD-tree for efficiency
        List<Set<Integer>> neighborLists = computeNeighborsParallel(X);

        // Find core points
        boolean[] isCore = new boolean[nSamples];
        for (int i = 0; i < nSamples; i++) {
            if (neighborLists.get(i).size() >= minSamples) {
                isCore[i] = true;
                coreSampleIndices.add(i);
            }
        }

        // Expand clusters
        for (int i = 0; i < nSamples; i++) {
            if (labels[i] != -1 || !isCore[i])
                continue;

            // Start new cluster
            expandCluster(X, neighborLists, i, clusterId, isCore);
            clusterId++;
        }

        nClusters = clusterId;
    }

    /**
     * Compute neighbor lists using spatial indexing for O(n log n) complexity.
     */
    private List<Set<Integer>> computeNeighborsParallel(float[][] X) {
        int nSamples = X.length;
        List<Set<Integer>> neighborLists = new ArrayList<>(nSamples);
        for (int i = 0; i < nSamples; i++) {
            neighborLists.add(ConcurrentHashMap.newKeySet());
        }

        // Build KD-tree for efficient range queries
        KDTree kdTree = new KDTree(X);

        // Parallel range queries
        IntStream.range(0, nSamples).parallel().forEach(i -> {
            int[] neighbors = kdTree.rangeQuery(X[i], eps);
            Set<Integer> neighborSet = neighborLists.get(i);
            for (int neighbor : neighbors) {
                neighborSet.add(neighbor);
                // Add symmetric relationship
                neighborLists.get(neighbor).add(i);
            }
        });

        return neighborLists;
    }

    /**
     * Expand cluster using BFS/DFS.
     */
    private void expandCluster(float[][] X, List<Set<Integer>> neighbors,
            int point, int clusterId, boolean[] isCore) {
        Queue<Integer> queue = new ArrayDeque<>();
        queue.add(point);
        labels[point] = clusterId;

        while (!queue.isEmpty()) {
            int current = queue.poll();

            if (!isCore[current])
                continue;

            for (int neighbor : neighbors.get(current)) {
                if (labels[neighbor] == -1) {
                    labels[neighbor] = clusterId;
                    queue.add(neighbor);
                }
            }
        }
    }

    public int[] predict(float[][] X) {
        // For DBSCAN, predict assigns to existing clusters or marks as noise
        int[] predictions = new int[X.length];

        // Simple approach: find nearest core point within eps
        for (int i = 0; i < X.length; i++) {
            predictions[i] = -1;
            double minDist = eps;

            for (int coreIdx : coreSampleIndices) {
                double dist = distance(X[i], X[coreIdx]);
                if (dist < minDist) {
                    minDist = dist;
                    predictions[i] = labels[coreIdx];
                }
            }
        }

        return predictions;
    }

    private double distance(float[] a, float[] b) {
        switch (metric) {
            case "manhattan":
                double sum = 0;
                for (int i = 0; i < a.length; i++) {
                    sum += Math.abs(a[i] - b[i]);
                }
                return sum;
            case "cosine":
                double dot = 0, normA = 0, normB = 0;
                for (int i = 0; i < a.length; i++) {
                    dot += a[i] * b[i];
                    normA += a[i] * a[i];
                    normB += b[i] * b[i];
                }
                return 1 - dot / (Math.sqrt(normA) * Math.sqrt(normB));
            default: // euclidean
                double sumSq = 0;
                for (int i = 0; i < a.length; i++) {
                    double diff = a[i] - b[i];
                    sumSq += diff * diff;
                }
                return Math.sqrt(sumSq);
        }
    }

    public int[] getLabels() {
        return labels;
    }

    public int getNClusters() {
        return nClusters;
    }

    public List<Integer> getCoreSampleIndices() {
        return coreSampleIndices;
    }

    /**
     * Simple KD-Tree implementation for efficient range queries.
     */
    private static class KDTree {
        private final Node root;
        private final float[][] points;

        private static class Node {
            float[] point;
            int index;
            Node left, right;
            int axis;
        }

        KDTree(float[][] points) {
            this.points = points;
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < points.length; i++)
                indices.add(i);
            this.root = buildTree(indices, 0);
        }

        private Node buildTree(List<Integer> indices, int depth) {
            if (indices.isEmpty())
                return null;

            int axis = depth % points[0].length;
            indices.sort((a, b) -> Float.compare(points[a][axis], points[b][axis]));

            int median = indices.size() / 2;
            Node node = new Node();
            node.index = indices.get(median);
            node.point = points[node.index];
            node.axis = axis;

            node.left = buildTree(indices.subList(0, median), depth + 1);
            node.right = buildTree(indices.subList(median + 1, indices.size()), depth + 1);

            return node;
        }

        int[] rangeQuery(float[] center, double radius) {
            List<Integer> results = new ArrayList<>();
            rangeQuery(root, center, radius * radius, results);
            return results.stream().mapToInt(i -> i).toArray();
        }

        private void rangeQuery(Node node, float[] center, double radiusSq, List<Integer> results) {
            if (node == null)
                return;

            // Check current point
            double distSq = 0;
            for (int i = 0; i < center.length; i++) {
                double diff = center[i] - node.point[i];
                distSq += diff * diff;
            }
            if (distSq <= radiusSq) {
                results.add(node.index);
            }

            // Check which side(s) to explore
            double axisDiff = center[node.axis] - node.point[node.axis];

            if (axisDiff <= 0) {
                rangeQuery(node.left, center, radiusSq, results);
                if (axisDiff * axisDiff <= radiusSq) {
                    rangeQuery(node.right, center, radiusSq, results);
                }
            } else {
                rangeQuery(node.right, center, radiusSq, results);
                if (axisDiff * axisDiff <= radiusSq) {
                    rangeQuery(node.left, center, radiusSq, results);
                }
            }
        }
    }
}