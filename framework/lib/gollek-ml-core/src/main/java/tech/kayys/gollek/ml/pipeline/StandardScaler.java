package tech.kayys.gollek.ml.pipeline;

import tech.kayys.gollek.ml.base.BaseTransformer;

/**
 * Transformers for feature preprocessing and engineering.
 */
public class StandardScaler extends BaseTransformer {
    private double[] mean;
    private double[] std;

    @Override
    public void fit(float[][] X) {
        int nSamples = X.length;
        int nFeatures = X[0].length;

        mean = new double[nFeatures];
        std = new double[nFeatures];

        // Compute mean
        for (float[] row : X) {
            for (int j = 0; j < nFeatures; j++) {
                mean[j] += row[j];
            }
        }
        for (int j = 0; j < nFeatures; j++) {
            mean[j] /= nSamples;
        }

        // Compute std
        for (float[] row : X) {
            for (int j = 0; j < nFeatures; j++) {
                double diff = row[j] - mean[j];
                std[j] += diff * diff;
            }
        }
        for (int j = 0; j < nFeatures; j++) {
            std[j] = Math.sqrt(std[j] / (nSamples - 1));
            if (std[j] < 1e-8)
                std[j] = 1;
        }
    }

    @Override
    public float[][] transform(float[][] X) {
        float[][] transformed = new float[X.length][];
        for (int i = 0; i < X.length; i++) {
            transformed[i] = new float[X[i].length];
            for (int j = 0; j < X[i].length; j++) {
                transformed[i][j] = (float) ((X[i][j] - mean[j]) / std[j]);
            }
        }
        return transformed;
    }

    @Override
    public float[] transformSingle(float[] x) {
        float[] transformed = new float[x.length];
        for (int j = 0; j < x.length; j++) {
            transformed[j] = (float) ((x[j] - mean[j]) / std[j]);
        }
        return transformed;
    }

    public void setParameters(double[] mean, double[] std) {
        this.mean = mean;
        this.std = std;
        this.setFitted(true);
    }
}
