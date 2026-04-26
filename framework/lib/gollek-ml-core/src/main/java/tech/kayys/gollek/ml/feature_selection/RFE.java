package tech.kayys.gollek.ml.feature_selection;

import tech.kayys.gollek.ml.base.*;
import tech.kayys.gollek.ml.linear_model.LinearModel;
import tech.kayys.gollek.ml.svm.SVC;
import tech.kayys.gollek.ml.ensemble.RandomForestClassifier;
import java.util.*;

/**
 * Recursive Feature Elimination (RFE).
 */
public class RFE extends BaseTransformer {
    private final BaseEstimator estimator;
    private final int nFeaturesToSelect;
    private final double step;
    private int[] selectedIndices;
    private List<Integer> ranking;

    public RFE(BaseEstimator estimator, int nFeaturesToSelect) {
        this(estimator, nFeaturesToSelect, 1.0);
    }

    public RFE(BaseEstimator estimator, int nFeaturesToSelect, double step) {
        this.estimator = estimator;
        this.nFeaturesToSelect = nFeaturesToSelect;
        this.step = step;
    }

    @Override
    public void fit(float[][] X) {
        throw new UnsupportedOperationException("RFE requires labels for supervised feature selection");
    }

    @Override
    public void fit(float[][] X, int[] y) {
        validateData(X, y);
        int nFeatures = X[0].length;
        List<Integer> remaining = new ArrayList<>();
        for (int i = 0; i < nFeatures; i++) remaining.add(i);

        ranking = new ArrayList<>(Collections.nCopies(nFeatures, 0));
        int rank = 1;

        float[][] currentX = X;

        while (remaining.size() > nFeaturesToSelect) {
            int nToRemove = Math.max(1, (int) (remaining.size() * step));
            if (remaining.size() - nToRemove < nFeaturesToSelect) {
                nToRemove = remaining.size() - nFeaturesToSelect;
            }

            // Get feature importances from the estimator
            estimator.fit(currentX, y);
            double[] importances = getFeatureImportances(estimator);

            // Find least important features
            List<Integer> sortedIndices = new ArrayList<>();
            for (int i = 0; i < remaining.size(); i++) sortedIndices.add(i);
            
            sortedIndices.sort((a, b) -> Double.compare(importances[a], importances[b]));

            List<Integer> toRemove = new ArrayList<>();
            for (int i = 0; i < nToRemove; i++) {
                int localIdx = sortedIndices.get(i);
                int globalIdx = remaining.get(localIdx);
                ranking.set(globalIdx, rank);
                toRemove.add(globalIdx);
            }

            for (Integer idx : toRemove) remaining.remove(idx);
            rank++;

            // Update currentX
            currentX = new float[X.length][remaining.size()];
            for (int i = 0; i < X.length; i++) {
                for (int j = 0; j < remaining.size(); j++) {
                    currentX[i][j] = X[i][remaining.get(j)];
                }
            }
        }

        selectedIndices = remaining.stream().mapToInt(i -> i).toArray();
        setFitted(true);
    }

    private double[] getFeatureImportances(BaseEstimator est) {
        if (est instanceof RandomForestClassifier) {
            return ((RandomForestClassifier) est).featureImportances();
        } else if (est instanceof LinearModel) {
            double[] coef = ((LinearModel) est).getCoefficients();
            double[] absCoef = new double[coef.length];
            for (int i = 0; i < coef.length; i++) absCoef[i] = Math.abs(coef[i]);
            return absCoef;
        } else if (est instanceof SVC) {
            // Placeholder for SVC importance
            return new double[0];
        }
        throw new UnsupportedOperationException("Estimator " + est.getClass().getSimpleName() + " does not provide feature importances");
    }

    @Override
    public float[][] transform(float[][] X) {
        validateInput(X);
        float[][] transformed = new float[X.length][selectedIndices.length];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < selectedIndices.length; j++) {
                transformed[i][j] = X[i][selectedIndices[j]];
            }
        }
        return transformed;
    }

    public int[] getRanking() {
        return ranking.stream().mapToInt(i -> i).toArray();
    }

    public int[] getSupport() {
        int[] support = new int[ranking.size()];
        for (int i = 0; i < ranking.size(); i++) {
            support[i] = ranking.get(i) == 0 ? 1 : 0;
        }
        return support;
    }
}
