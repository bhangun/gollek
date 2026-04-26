package tech.kayys.gollek.ml.feature_selection;

import tech.kayys.gollek.ml.base.*;
import java.util.*;
import java.util.stream.IntStream;

/**
 * Mutual Information for feature selection.
 */
public class MutualInfo extends BaseTransformer {
    private double[] scores;
    private int[] selectedIndices;
    private final int nFeaturesToSelect;

    public MutualInfo(int nFeaturesToSelect) {
        this.nFeaturesToSelect = nFeaturesToSelect;
    }

    @Override
    public void fit(float[][] X) {
        int nFeatures = X[0].length;
        scores = new double[nFeatures];

        // This is unsupervised MI - would typically need labels for supervised
        // But if labels are not provided, we use self-information or dummy labels
        // However, the base class has fit(X, y).
    }

    @Override
    public void fit(float[][] X, int[] y) {
        validateData(X, y);
        int nFeatures = X[0].length;
        scores = new double[nFeatures];

        // Calculate mutual information for each feature
        IntStream.range(0, nFeatures).parallel().forEach(i -> {
            scores[i] = mutualInformation(X, y, i);
        });

        // Select top features
        Integer[] indices = new Integer[nFeatures];
        for (int i = 0; i < nFeatures; i++)
            indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(scores[b], scores[a]));

        int nToSelect = Math.min(nFeaturesToSelect, nFeatures);
        selectedIndices = new int[nToSelect];
        for (int i = 0; i < nToSelect; i++) {
            selectedIndices[i] = indices[i];
        }
        setFitted(true);
    }

    private double mutualInformation(float[][] X, int[] y, int featureIdx) {
        float[] values = new float[X.length];
        for (int i = 0; i < X.length; i++) {
            values[i] = X[i][featureIdx];
        }

        int nBins = 10;
        float min = values[0];
        float max = values[0];
        for (float v : values) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        float step = (max - min) / nBins;
        if (step == 0) step = 1;

        int nClasses = 0;
        for (int val : y) if (val >= nClasses) nClasses = val + 1;

        int[][] joint = new int[nBins][nClasses];
        int[] marginalX = new int[nBins];
        int[] marginalY = new int[nClasses];

        for (int i = 0; i < X.length; i++) {
            int binX = Math.min(nBins - 1, (int) ((values[i] - min) / step));
            int binY = y[i];
            joint[binX][binY]++;
            marginalX[binX]++;
            marginalY[binY]++;
        }

        double mi = 0;
        int total = X.length;
        for (int i = 0; i < nBins; i++) {
            for (int j = 0; j < nClasses; j++) {
                if (joint[i][j] > 0) {
                    double pxy = joint[i][j] / (double) total;
                    double px = marginalX[i] / (double) total;
                    double py = marginalY[j] / (double) total;
                    mi += pxy * Math.log(pxy / (px * py + 1e-11) + 1e-11);
                }
            }
        }

        return mi;
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

    public double[] getScores() {
        return scores;
    }

    public int[] getSelectedIndices() {
        return selectedIndices;
    }
}