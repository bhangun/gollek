package tech.kayys.gollek.ml.util;

import tech.kayys.gollek.ml.base.BaseEstimator;
import tech.kayys.gollek.ml.model_selection.ModelSelection;
import tech.kayys.gollek.ml.metrics.Metrics;
import java.util.*;
import java.util.stream.*;

/**
 * Cross-validation utilities.
 */
public class CrossValidation {

    /**
     * K-Fold cross-validation score.
     */
    public static double crossValScore(BaseEstimator estimator, float[][] X, int[] y,
            int nFolds, String scoring) {
        ModelSelection.KFold kfold = new ModelSelection.KFold(nFolds);
        List<ModelSelection.Fold> folds = kfold.split(X.length);
        double[] scores = new double[nFolds];

        // Parallel execution of folds
        IntStream.range(0, nFolds).parallel().forEach(foldIdx -> {
            ModelSelection.Fold foldData = folds.get(foldIdx);

            // Create a fresh copy of the estimator
            BaseEstimator foldEstimator = estimator.clone();

            // Extract training and validation sets
            float[][] XTrain = new float[foldData.trainIndices.length][];
            int[] yTrain = new int[foldData.trainIndices.length];
            for (int i = 0; i < foldData.trainIndices.length; i++) {
                XTrain[i] = X[foldData.trainIndices[i]];
                yTrain[i] = y[foldData.trainIndices[i]];
            }

            float[][] XVal = new float[foldData.valIndices.length][];
            int[] yVal = new int[foldData.valIndices.length];
            for (int i = 0; i < foldData.valIndices.length; i++) {
                XVal[i] = X[foldData.valIndices[i]];
                yVal[i] = y[foldData.valIndices[i]];
            }

            // Train and evaluate
            foldEstimator.fit(XTrain, yTrain);

            if ("accuracy".equals(scoring)) {
                scores[foldIdx] = foldEstimator.score(XVal, yVal);
            } else if ("f1".equals(scoring)) {
                int[] predictions = foldEstimator.predict(XVal);
                scores[foldIdx] = Metrics.f1Score(yVal, predictions, 1); // Assuming binary class 1 for now
            }
        });

        return Arrays.stream(scores).average().orElse(0.0);
    }

    /**
     * Grid search for hyperparameter tuning.
     */
    public static GridSearchResult gridSearch(BaseEstimator estimator,
            Map<String, Object[]> paramGrid,
            float[][] X, int[] y, int nFolds) {
        List<Map<String, Object>> paramSets = generateParameterSets(paramGrid);

        // Evaluate each parameter set in parallel
        List<GridSearchResult> results = paramSets.parallelStream()
                .map(params -> {
                    BaseEstimator clone = estimator.clone();
                    clone.setParams(params);
                    double score = crossValScore(clone, X, y, nFolds, "accuracy");
                    return new GridSearchResult(params, score);
                })
                .collect(Collectors.toList());

        // Find best parameters
        GridSearchResult best = results.stream()
                .max((a, b) -> Double.compare(a.score, b.score))
                .orElse(null);

        // Train best model on all data
        if (best != null) {
            estimator.setParams(best.params);
            estimator.fit(X, y);
        }

        return best;
    }

    private static List<Map<String, Object>> generateParameterSets(Map<String, Object[]> paramGrid) {
        List<Map<String, Object>> combinations = new ArrayList<>();
        combinations.add(new LinkedHashMap<>());

        for (Map.Entry<String, Object[]> entry : paramGrid.entrySet()) {
            String key = entry.getKey();
            Object[] values = entry.getValue();
            List<Map<String, Object>> nextCombinations = new ArrayList<>();

            for (Map<String, Object> combination : combinations) {
                for (Object value : values) {
                    Map<String, Object> next = new LinkedHashMap<>(combination);
                    next.put(key, value);
                    nextCombinations.add(next);
                }
            }
            combinations = nextCombinations;
        }
        return combinations;
    }

    // Helper classes
    public static class GridSearchResult {
        public Map<String, Object> params;
        public double score;

        public GridSearchResult(Map<String, Object> params, double score) {
            this.params = params;
            this.score = score;
        }
    }

    public static class LearningCurve {
        public List<Integer> trainSizes = new ArrayList<>();
        public List<Double> trainScores = new ArrayList<>();
        public List<Double> trainStd = new ArrayList<>();
    }
}