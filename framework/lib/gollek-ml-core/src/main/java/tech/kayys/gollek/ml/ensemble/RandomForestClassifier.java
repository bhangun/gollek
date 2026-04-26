package tech.kayys.gollek.ml.ensemble;

import tech.kayys.gollek.ml.base.BaseEstimator;
import tech.kayys.gollek.ml.tree.DecisionTreeClassifier;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.IntStream;

/**
 * Random Forest - bagging ensemble of decision trees.
 * Supports parallel tree building and out-of-bag scoring.
 */
public class RandomForestClassifier extends BaseEstimator {

    private final List<DecisionTreeClassifier> trees = new ArrayList<>();
    private final int nEstimators;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final String criterion;
    private final String maxFeatures;
    private final boolean bootstrap;
    private final int nJobs;
    private final boolean oobScore;

    private int nClasses;
    private int nFeatures;
    private double oobScore_;
    private Object[] oobPredictions;
    private int[] oobCounts;
    private boolean isFitted = false;

    public RandomForestClassifier() {
        this(100, 10, 2, "gini", "sqrt", true, -1, false);
    }

    public RandomForestClassifier(int nEstimators, int maxDepth, int minSamplesSplit,
            String criterion, String maxFeatures, boolean bootstrap,
            int nJobs, boolean oobScore) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.criterion = criterion;
        this.maxFeatures = maxFeatures;
        this.bootstrap = bootstrap;
        this.nJobs = nJobs > 0 ? nJobs : Runtime.getRuntime().availableProcessors();
        this.oobScore = oobScore;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        validateData(X, y);
        int nSamples = X.length;
        this.nFeatures = X[0].length;
        this.nClasses = (int) Arrays.stream(y).distinct().count();

        // Initialize OOB tracking if needed
        if (oobScore) {
            oobPredictions = new Object[nSamples];
            oobCounts = new int[nSamples];
            for (int i = 0; i < nSamples; i++) {
                oobPredictions[i] = new int[nClasses];
            }
        }

        // Build trees in parallel
        ExecutorService executor = Executors.newFixedThreadPool(nJobs);
        List<Future<DecisionTreeClassifier>> futures = new ArrayList<>();

        for (int i = 0; i < nEstimators; i++) {
            final int seed = i;
            futures.add(executor.submit(() -> {
                DecisionTreeClassifier tree = new DecisionTreeClassifier(
                        maxDepth, minSamplesSplit, 1, criterion, maxFeatures);

                // Bootstrap sampling
                int[] indices;
                int[] oobIndices = null;

                if (bootstrap) {
                    indices = bootstrapSample(nSamples, seed);
                    if (oobScore) {
                        oobIndices = findOobIndices(indices, nSamples);
                    }
                } else {
                    indices = new int[nSamples];
                    for (int j = 0; j < nSamples; j++)
                        indices[j] = j;
                }

                // Extract bootstrap sample
                float[][] Xboot = new float[indices.length][];
                int[] yboot = new int[indices.length];
                for (int j = 0; j < indices.length; j++) {
                    Xboot[j] = X[indices[j]];
                    yboot[j] = y[indices[j]];
                }

                // Train tree
                tree.fit(Xboot, yboot);

                // Update OOB predictions
                if (oobScore && oobIndices != null) {
                    for (int idx : oobIndices) {
                        int pred = tree.predictSingle(X[idx]);
                        synchronized (oobPredictions[idx]) {
                            ((int[]) oobPredictions[idx])[pred]++;
                            oobCounts[idx]++;
                        }
                    }
                }

                return tree;
            }));
        }

        // Collect results
        for (Future<DecisionTreeClassifier> future : futures) {
            try {
                trees.add(future.get());
            } catch (Exception e) {
                throw new RuntimeException("Tree building failed", e);
            }
        }
        executor.shutdown();

        // Calculate OOB score if requested
        if (oobScore) {
            int correct = 0;
            int total = 0;
            for (int i = 0; i < nSamples; i++) {
                if (oobCounts[i] > 0) {
                    int majority = argmax((int[]) oobPredictions[i]);
                    if (majority == y[i])
                        correct++;
                    total++;
                }
            }
            oobScore_ = total > 0 ? (double) correct / total : 0.0;
        }
        
        this.isFitted = true;
    }

    /**
     * Bootstrap sampling with replacement.
     */
    private int[] bootstrapSample(int nSamples, int seed) {
        Random rng = new Random(seed);
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            indices[i] = rng.nextInt(nSamples);
        }
        return indices;
    }

    /**
     * Find OOB indices (samples not in bootstrap sample).
     */
    private int[] findOobIndices(int[] bootstrapIndices, int nSamples) {
        boolean[] selected = new boolean[nSamples];
        for (int idx : bootstrapIndices) {
            selected[idx] = true;
        }

        List<Integer> oobList = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            if (!selected[i])
                oobList.add(i);
        }

        return oobList.stream().mapToInt(i -> i).toArray();
    }

    /**
     * Predict by majority voting.
     */
    @Override
    public int[] predict(float[][] X) {
        validateInput(X);
        int[] predictions = new int[X.length];

        // Parallel prediction
        IntStream.range(0, X.length).parallel().forEach(i -> {
            int[] votes = new int[nClasses];
            for (DecisionTreeClassifier tree : trees) {
                votes[tree.predictSingle(X[i])]++;
            }
            predictions[i] = argmax(votes);
        });

        return predictions;
    }

    /**
     * Predict probabilities (averaged across trees).
     */
    @Override
    public double[][] predictProba(float[][] X) {
        validateInput(X);
        double[][] probabilities = new double[X.length][nClasses];

        IntStream.range(0, X.length).parallel().forEach(i -> {
            for (DecisionTreeClassifier tree : trees) {
                double[][] treeProbs = tree.predictProba(new float[][] { X[i] });
                for (int c = 0; c < nClasses; c++) {
                    probabilities[i][c] += treeProbs[0][c];
                }
            }
            for (int c = 0; c < nClasses; c++) {
                probabilities[i][c] /= trees.size();
            }
        });

        return probabilities;
    }

    public double oobScore() {
        return oobScore_;
    }

    @Override
    public boolean isFitted() {
        return isFitted;
    }

    private int argmax(int[] values) {
        int bestIdx = 0;
        int maxVal = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > maxVal) {
                maxVal = values[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    public double[] featureImportances() {
        double[] importances = new double[nFeatures];
        for (DecisionTreeClassifier tree : trees) {
            double[] treeImportances = tree.featureImportances();
            for (int i = 0; i < nFeatures; i++) {
                importances[i] += treeImportances[i];
            }
        }
        for (int i = 0; i < nFeatures; i++) {
            importances[i] /= trees.size();
        }
        return importances;
    }
}