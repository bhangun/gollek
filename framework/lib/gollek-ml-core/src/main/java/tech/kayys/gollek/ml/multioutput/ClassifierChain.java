package tech.kayys.gollek.ml.multioutput;

import tech.kayys.gollek.ml.base.*;
import java.util.*;

/**
 * Classifier chain for multi-label classification.
 */
public class ClassifierChain extends BaseEstimator {
    private final BaseEstimator baseEstimator;
    private final String order; // "random" or "feature"
    private List<BaseEstimator> chain;
    private int[] labelOrder;

    public ClassifierChain(BaseEstimator baseEstimator) {
        this(baseEstimator, "feature");
    }

    public ClassifierChain(BaseEstimator baseEstimator, String order) {
        this.baseEstimator = baseEstimator;
        this.order = order;
    }

    @Override
    public void fit(float[][] X, int[] y) {
        fit(X, new int[][] { y });
    }

    @Override
    public void fit(float[][] X, int[][] y) {
        validateData(X, y);
        int nLabels = y[0].length;
        chain = new ArrayList<>();

        // Determine label order
        labelOrder = new int[nLabels];
        for (int i = 0; i < nLabels; i++)
            labelOrder[i] = i;

        if ("random".equals(order)) {
            Random rng = new Random(42);
            for (int i = nLabels - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int temp = labelOrder[i];
                labelOrder[i] = labelOrder[j];
                labelOrder[j] = temp;
            }
        }

        // Train chain
        float[][] currentX = X;
        for (int i = 0; i < nLabels; i++) {
            int label = labelOrder[i];

            // Extract labels for this position
            int[] yLabel = new int[y.length];
            for (int j = 0; j < y.length; j++) {
                yLabel[j] = y[j][label];
            }

            // Train classifier
            BaseEstimator classifier = baseEstimator.clone();
            classifier.fit(currentX, yLabel);
            chain.add(classifier);

            // Append predictions as new features
            int[] predictions = classifier.predict(currentX);
            currentX = augmentFeatures(currentX, predictions);
        }

        setFitted(true);
    }

    @Override
    public int[] predict(float[][] X) {
        throw new UnsupportedOperationException("Use predictMulti for multi-label");
    }

    public int[][] predictMulti(float[][] X) {
        validateInput(X);
        int nLabels = chain.size();
        int[][] predictions = new int[X.length][nLabels];

        float[][] currentX = X;
        for (int i = 0; i < nLabels; i++) {
            int[] pred = chain.get(i).predict(currentX);
            int label = labelOrder[i];
            for (int j = 0; j < X.length; j++) {
                predictions[j][label] = pred[j];
            }
            currentX = augmentFeatures(currentX, pred);
        }

        return predictions;
    }

    private float[][] augmentFeatures(float[][] X, int[] newFeature) {
        float[][] augmented = new float[X.length][X[0].length + 1];
        for (int i = 0; i < X.length; i++) {
            System.arraycopy(X[i], 0, augmented[i], 0, X[i].length);
            augmented[i][X[i].length] = newFeature[i];
        }
        return augmented;
    }
}
