package tech.kayys.gollek.ml.clustering;

import tech.kayys.gollek.ml.base.BaseEstimator;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.stream.IntStream;

/**
 * K-Means clustering with K-Means++ initialization and parallel execution.
 */
public class KMeans extends BaseEstimator {
    private final int nClusters;
    private final int maxIterations;
    private final double tolerance;
    private final int nInit;
    private final String init; // "k-means++" or "random"
    private final int nJobs;

    private double[][] centers;
    private int[] labels;
    private double inertia;
    private int nIterations;

    public KMeans(int nClusters) {
        this(nClusters, 300, 1e-4, 10, "k-means++", -1);
    }

    public KMeans(int nClusters, int maxIterations, double tolerance,
            int nInit, String init, int nJobs) {
        this.nClusters = nClusters;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.nInit = nInit;
        this.init = init;
        this.nJobs = nJobs > 0 ? nJobs : Runtime.getRuntime().availableProcessors();
    }

    @Override
    public void fit(float[][] X, int[] y) {
        fit(X);
    }

    public void fit(float[][] X) {
        int nSamples = X.length;
        int nFeatures = X[0].length;

        // Try multiple initializations and pick the best
        double bestInertia = Double.MAX_VALUE;
        double[][] bestCenters = null;
        int[] bestLabels = null;

        ExecutorService executor = Executors.newFixedThreadPool(nJobs);
        List<Future<InitResult>> futures = new ArrayList<>();

        for (int initRun = 0; initRun < nInit; initRun++) {
            final int run = initRun;
            futures.add(executor.submit(() -> {
                double[][] initialCenters = initializeCenters(X, run);
                return runKMeans(X, initialCenters);
            }));
        }

        for (Future<InitResult> future : futures) {
            try {
                InitResult result = future.get();
                if (result.inertia < bestInertia) {
                    bestInertia = result.inertia;
                    bestCenters = result.centers;
                    bestLabels = result.labels;
                    nIterations = result.iterations;
                }
            } catch (Exception e) {
                throw new RuntimeException("K-Means initialization failed", e);
            }
        }
        executor.shutdown();

        this.centers = bestCenters;
        this.labels = bestLabels;
        this.inertia = bestInertia;
    }

    /**
     * K-Means++ initialization for better centroid placement.
     */
    private double[][] initializeCenters(float[][] X, int seed) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        Random rng = new Random(seed);

        double[][] centers = new double[nClusters][nFeatures];

        // First center: random sample
        int firstIdx = rng.nextInt(nSamples);
        centers[0] = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++) {
            centers[0][j] = X[firstIdx][j];
        }

        if ("k-means++".equals(init)) {
            // K-Means++: choose remaining centers with probability proportional to distance
            // squared
            double[] minDistSq = new double[nSamples];
            Arrays.fill(minDistSq, Double.MAX_VALUE);

            for (int c = 1; c < nClusters; c++) {
                // Update distances to nearest center
                for (int i = 0; i < nSamples; i++) {
                    double distSq = squaredDistance(X[i], centers[c - 1]);
                    if (distSq < minDistSq[i]) {
                        minDistSq[i] = distSq;
                    }
                }

                // Choose next center with probability proportional to distance squared
                double sumDist = 0;
                for (double d : minDistSq)
                    sumDist += d;

                double threshold = rng.nextDouble() * sumDist;
                double cumulative = 0;
                int nextIdx = 0;
                for (int i = 0; i < nSamples; i++) {
                    cumulative += minDistSq[i];
                    if (cumulative >= threshold) {
                        nextIdx = i;
                        break;
                    }
                }

                centers[c] = new double[nFeatures];
                for (int j = 0; j < nFeatures; j++) {
                    centers[c][j] = X[nextIdx][j];
                }
            }
        } else {
            // Random initialization
            for (int c = 1; c < nClusters; c++) {
                int idx = rng.nextInt(nSamples);
                centers[c] = new double[nFeatures];
                for (int j = 0; j < nFeatures; j++) {
                    centers[c][j] = X[idx][j];
                }
            }
        }

        return centers;
    }

    /**
     * Run K-Means algorithm until convergence.
     */
    private InitResult runKMeans(float[][] X, double[][] initialCenters) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        double[][] centers = copyCenters(initialCenters);
        int[] labels = new int[nSamples];
        double prevInertia = Double.MAX_VALUE;

        for (int iter = 0; iter < maxIterations; iter++) {
            // Assign samples to nearest center
            assignClusters(X, centers, labels);

            // Update centers
            double[][] newCenters = new double[nClusters][nFeatures];
            int[] counts = new int[nClusters];

            for (int i = 0; i < nSamples; i++) {
                int cluster = labels[i];
                counts[cluster]++;
                for (int j = 0; j < nFeatures; j++) {
                    newCenters[cluster][j] += X[i][j];
                }
            }

            // Handle empty clusters
            for (int c = 0; c < nClusters; c++) {
                if (counts[c] > 0) {
                    for (int j = 0; j < nFeatures; j++) {
                        newCenters[c][j] /= counts[c];
                    }
                } else {
                    // Re-initialize empty cluster
                    int randomIdx = ThreadLocalRandom.current().nextInt(nSamples);
                    newCenters[c] = new double[nFeatures];
                    for (int j = 0; j < nFeatures; j++) {
                        newCenters[c][j] = X[randomIdx][j];
                    }
                }
            }

            // Compute inertia
            double inertia = computeInertia(X, labels, newCenters);

            // Check convergence
            if (Math.abs(prevInertia - inertia) < tolerance) {
                return new InitResult(newCenters, labels, inertia, iter + 1);
            }

            prevInertia = inertia;
            centers = newCenters;
        }

        return new InitResult(centers, labels, prevInertia, maxIterations);
    }

    /**
     * Assign clusters in parallel using SIMD distance calculations.
     */
    private void assignClusters(float[][] X, double[][] centers, int[] labels) {
        IntStream.range(0, X.length).parallel().forEach(i -> {
            float[] point = X[i];
            int bestCluster = 0;
            double bestDist = squaredDistance(point, centers[0]);

            for (int c = 1; c < nClusters; c++) {
                double dist = squaredDistance(point, centers[c]);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestCluster = c;
                }
            }
            labels[i] = bestCluster;
        });
    }

    private double squaredDistance(float[] point, double[] center) {
        double sum = 0;
        for (int j = 0; j < point.length; j++) {
            double diff = point[j] - center[j];
            sum += diff * diff;
        }
        return sum;
    }

    private double computeInertia(float[][] X, int[] labels, double[][] centers) {
        double inertia = 0;
        for (int i = 0; i < X.length; i++) {
            inertia += squaredDistance(X[i], centers[labels[i]]);
        }
        return inertia;
    }

    private double[][] copyCenters(double[][] centers) {
        double[][] copy = new double[centers.length][];
        for (int i = 0; i < centers.length; i++) {
            copy[i] = centers[i].clone();
        }
        return copy;
    }

    public int[] predict(float[][] X) {
        int[] predictions = new int[X.length];
        assignClusters(X, centers, predictions);
        return predictions;
    }

    public double[][] transform(float[][] X) {
        double[][] distances = new double[X.length][nClusters];
        IntStream.range(0, X.length).parallel().forEach(i -> {
            for (int c = 0; c < nClusters; c++) {
                distances[i][c] = Math.sqrt(squaredDistance(X[i], centers[c]));
            }
        });
        return distances;
    }

    public double[][] getCenters() {
        return centers;
    }

    public int[] getLabels() {
        return labels;
    }

    public double getInertia() {
        return inertia;
    }

    public int getIterations() {
        return nIterations;
    }

    private static class InitResult {
        final double[][] centers;
        final int[] labels;
        final double inertia;
        final int iterations;

        InitResult(double[][] centers, int[] labels, double inertia, int iterations) {
            this.centers = centers;
            this.labels = labels;
            this.inertia = inertia;
            this.iterations = iterations;
        }
    }
}