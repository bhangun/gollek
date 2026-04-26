package tech.kayys.gollek.ml.pipeline;

import tech.kayys.gollek.ml.base.BaseTransformer;

/**
 * Principal Component Analysis (PCA).
 */
public class PCA extends BaseTransformer {
    private final int nComponents;
    private double[] explainedVariance;
    private double[][] components;
    private double[] mean;

    public PCA(int nComponents) {
        this.nComponents = nComponents;
    }

    @Override
    public void fit(float[][] X) {
        int nSamples = X.length;
        int nFeatures = X[0].length;

        // Center the data
        mean = new double[nFeatures];
        for (float[] row : X) {
            for (int j = 0; j < nFeatures; j++) {
                mean[j] += row[j];
            }
        }
        for (int j = 0; j < nFeatures; j++) {
            mean[j] /= nSamples;
        }

        double[][] centered = new double[nSamples][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                centered[i][j] = X[i][j] - mean[j];
            }
        }

        // Compute covariance matrix
        double[][] cov = new double[nFeatures][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                for (int k = 0; k < nFeatures; k++) {
                    cov[j][k] += centered[i][j] * centered[i][k];
                }
            }
        }
        for (int j = 0; j < nFeatures; j++) {
            for (int k = 0; k < nFeatures; k++) {
                cov[j][k] /= (nSamples - 1);
            }
        }

        // Power iteration for top eigenvectors
        components = powerIteration(cov, nComponents);

        // Compute explained variance
        explainedVariance = new double[nComponents];
        for (int i = 0; i < nComponents; i++) {
            double[] comp = components[i];
            double var = 0;
            for (int j = 0; j < nFeatures; j++) {
                for (int k = 0; k < nFeatures; k++) {
                    var += comp[j] * cov[j][k] * comp[k];
                }
            }
            explainedVariance[i] = var;
        }
    }

    private double[][] powerIteration(double[][] matrix, int k) {
        int n = matrix.length;
        double[][] components = new double[k][n];

        for (int i = 0; i < k; i++) {
            double[] eigenvector = new double[n];
            eigenvector[i % n] = 1.0;

            for (int iter = 0; iter < 100; iter++) {
                double[] next = new double[n];
                for (int j = 0; j < n; j++) {
                    for (int m = 0; m < n; m++) {
                        next[j] += matrix[j][m] * eigenvector[m];
                    }
                }

                // Normalize
                double norm = 0;
                for (int j = 0; j < n; j++)
                    norm += next[j] * next[j];
                norm = Math.sqrt(norm);
                for (int j = 0; j < n; j++)
                    next[j] /= norm;

                // Check convergence
                double diff = 0;
                for (int j = 0; j < n; j++)
                    diff += Math.abs(next[j] - eigenvector[j]);
                if (diff < 1e-6)
                    break;

                eigenvector = next;
            }

            components[i] = eigenvector;

            // Deflate matrix
            for (int j = 0; j < n; j++) {
                for (int m = 0; m < n; m++) {
                    matrix[j][m] -= eigenvector[j] * eigenvector[m];
                }
            }
        }

        return components;
    }

    @Override
    public float[][] transform(float[][] X) {
        float[][] transformed = new float[X.length][nComponents];

        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < nComponents; j++) {
                double sum = 0;
                for (int k = 0; k < X[i].length; k++) {
                    sum += (X[i][k] - mean[k]) * components[j][k];
                }
                transformed[i][j] = (float) sum;
            }
        }

        return transformed;
    }

    public void setParameters(double[][] components, double[] mean, double[] explainedVariance) {
        this.components = components;
        this.mean = mean;
        this.explainedVariance = explainedVariance;
        this.setFitted(true);
    }
}
