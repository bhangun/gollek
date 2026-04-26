package tech.kayys.gollek.ml.model_selection;

import tech.kayys.gollek.ml.base.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import tech.kayys.gollek.ml.metrics.Metrics;

/**
 * Randomized search for hyperparameter optimization.
 * More efficient than grid search for high-dimensional parameter spaces.
 */
public class RandomizedSearchCV {
    private final BaseEstimator estimator;
    private final Map<String, Object[]> paramDistributions;
    private final int nIter;
    private final int cv;
    private final String scoring;
    private final int nJobs;
    private final int randomState;

    private double bestScore;
    private Map<String, Object> bestParams;
    private List<Map<String, Object>> cvResults;

    public RandomizedSearchCV(BaseEstimator estimator, Map<String, Object[]> paramDistributions,
            int nIter, int cv, String scoring, int nJobs, int randomState) {
        this.estimator = estimator;
        this.paramDistributions = paramDistributions;
        this.nIter = nIter;
        this.cv = cv;
        this.scoring = scoring;
        this.nJobs = nJobs;
        this.randomState = randomState;
        this.cvResults = new ArrayList<>();
    }

    public void fit(float[][] X, int[] y) {
        Random rng = new Random(randomState);
        ExecutorService executor = Executors
                .newFixedThreadPool(nJobs > 0 ? nJobs : Runtime.getRuntime().availableProcessors());

        List<Future<ParamResult>> futures = new ArrayList<>();

        for (int i = 0; i < nIter; i++) {
            Map<String, Object> params = sampleParameters(rng);
            futures.add(executor.submit(() -> evaluateParameters(X, y, params)));
        }

        bestScore = Double.NEGATIVE_INFINITY;
        for (Future<ParamResult> future : futures) {
            try {
                ParamResult result = future.get();
                cvResults.add(Map.of(
                        "params", result.params,
                        "mean_score", result.meanScore,
                        "std_score", result.stdScore,
                        "scores", result.scores));

                if (result.meanScore > bestScore) {
                    bestScore = result.meanScore;
                    bestParams = result.params;
                }
            } catch (Exception e) {
                throw new RuntimeException("Randomized search failed", e);
            }
        }
        executor.shutdown();

        // Train best model on all data
        estimator.setParams(bestParams);
        estimator.fit(X, y);
    }

    private Map<String, Object> sampleParameters(Random rng) {
        Map<String, Object> params = new LinkedHashMap<>();

        for (Map.Entry<String, Object[]> entry : paramDistributions.entrySet()) {
            Object[] values = entry.getValue();
            params.put(entry.getKey(), values[rng.nextInt(values.length)]);
        }

        return params;
    }

    private ParamResult evaluateParameters(float[][] X, int[] y, Map<String, Object> params) {
        BaseEstimator clone = estimator.clone();
        clone.setParams(params);

        ModelSelection.KFold kfold = new ModelSelection.KFold(cv, true, randomState + params.hashCode());
        List<ModelSelection.Fold> folds = kfold.split(X.length);

        double[] scores = new double[cv];

        for (int foldIdx = 0; foldIdx < folds.size(); foldIdx++) {
            ModelSelection.Fold fold = folds.get(foldIdx);

            // Split data
            float[][] XTrain = new float[fold.trainIndices.length][];
            int[] yTrain = new int[fold.trainIndices.length];
            float[][] XVal = new float[fold.valIndices.length][];
            int[] yVal = new int[fold.valIndices.length];

            for (int i = 0; i < fold.trainIndices.length; i++) {
                int idx = fold.trainIndices[i];
                XTrain[i] = X[idx];
                yTrain[i] = y[idx];
            }
            for (int i = 0; i < fold.valIndices.length; i++) {
                int idx = fold.valIndices[i];
                XVal[i] = X[idx];
                yVal[i] = y[idx];
            }

            // Train and evaluate
            BaseEstimator foldEstimator = clone.clone();
            foldEstimator.fit(XTrain, yTrain);

            if ("accuracy".equals(scoring)) {
                int[] predictions = foldEstimator.predict(XVal);
                int correct = 0;
                for (int i = 0; i < predictions.length; i++) {
                    if (predictions[i] == yVal[i])
                        correct++;
                }
                scores[foldIdx] = (double) correct / predictions.length;
            } else if ("f1".equals(scoring)) {
                int[] predictions = foldEstimator.predict(XVal);
                scores[foldIdx] = Metrics.f1Score(yVal, predictions);
            } else if ("roc_auc".equals(scoring)) {
                double[][] probs = foldEstimator.predictProba(XVal);
                float[] scoresForAUC = new float[yVal.length];
                for (int i = 0; i < yVal.length; i++) {
                    scoresForAUC[i] = (float) probs[i][1];
                }
                scores[foldIdx] = Metrics.rocAuc(scoresForAUC, yVal);
            }
        }

        double meanScore = Arrays.stream(scores).average().orElse(0);
        double stdScore = Math.sqrt(Arrays.stream(scores)
                .map(s -> Math.pow(s - meanScore, 2))
                .average().orElse(0));

        return new ParamResult(params, meanScore, stdScore, scores);
    }

    public double getBestScore() {
        return bestScore;
    }

    public Map<String, Object> getBestParams() {
        return bestParams;
    }

    public List<Map<String, Object>> getCvResults() {
        return cvResults;
    }

    private static class ParamResult {
        final Map<String, Object> params;
        final double meanScore;
        final double stdScore;
        final double[] scores;

        ParamResult(Map<String, Object> params, double meanScore, double stdScore, double[] scores) {
            this.params = params;
            this.meanScore = meanScore;
            this.stdScore = stdScore;
            this.scores = scores;
        }
    }
}