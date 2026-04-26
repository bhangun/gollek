package tech.kayys.gollek.ml.multioutput;

import tech.kayys.gollek.ml.base.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Multi-output classifier using one-vs-rest strategy.
 */
public class MultiOutputClassifier extends BaseEstimator {
    private final BaseEstimator estimator;
    private List<BaseEstimator> estimators;
    private int nOutputs;

    public MultiOutputClassifier(BaseEstimator estimator) {
        this.estimator = estimator;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        // Wrap single output into multi-output with 1 output
        fit(X, new int[][] { y });
    }

    @Override
    public void fit(float[][] X, int[][] y) {
        validateData(X, y);
        nOutputs = y[0].length;
        estimators = new ArrayList<>();

        // Train one estimator per output
        ExecutorService executor = Executors.newFixedThreadPool(
                Runtime.getRuntime().availableProcessors());

        List<Future<BaseEstimator>> futures = new ArrayList<>();

        for (int i = 0; i < nOutputs; i++) {
            final int outputIdx = i;
            futures.add(executor.submit(() -> {
                int[] ySingle = new int[y.length];
                for (int j = 0; j < y.length; j++) {
                    ySingle[j] = y[j][outputIdx];
                }
                BaseEstimator est = estimator.clone();
                est.fit(X, ySingle);
                return est;
            }));
        }

        for (Future<BaseEstimator> future : futures) {
            try {
                estimators.add(future.get());
            } catch (Exception e) {
                throw new RuntimeException("Multi-output training failed", e);
            }
        }
        executor.shutdown();

        setFitted(true);
    }

    @Override
    public int[] predict(float[][] X) {
        throw new UnsupportedOperationException("Use predictMulti for multi-output");
    }

    public int[][] predictMulti(float[][] X) {
        validateInput(X);
        int[][] predictions = new int[X.length][nOutputs];

        for (int i = 0; i < nOutputs; i++) {
            int[] pred = estimators.get(i).predict(X);
            for (int j = 0; j < X.length; j++) {
                predictions[j][i] = pred[j];
            }
        }

        return predictions;
    }

    @Override
    public double[][] predictProba(float[][] X) {
        validateInput(X);
        double[][] allProbs = new double[X.length][nOutputs * 2]; // Assuming binary
        int offset = 0;

        for (int i = 0; i < nOutputs; i++) {
            double[][] probs = estimators.get(i).predictProba(X);
            for (int j = 0; j < X.length; j++) {
                allProbs[j][offset] = probs[j][0];
                allProbs[j][offset + 1] = probs[j][1];
            }
            offset += 2;
        }

        return allProbs;
    }
}