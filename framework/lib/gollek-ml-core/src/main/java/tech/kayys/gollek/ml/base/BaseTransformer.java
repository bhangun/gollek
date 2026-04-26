package tech.kayys.gollek.ml.base;

import io.quarkus.runtime.annotations.RegisterForReflection;
import java.util.ArrayList;
import java.util.List;

/**
 * Base class for all transformers (preprocessing, feature extraction).
 * Extends BaseEstimator with transformation capabilities.
 */
@RegisterForReflection(registerFullHierarchy = true)
public abstract class BaseTransformer extends BaseEstimator {

    /**
     * Fit transformer to data (computes statistics like mean, std).
     */
    public abstract void fit(float[][] X);

    /**
     * Transform data using fitted parameters.
     */
    public abstract float[][] transform(float[][] X);

    /**
     * Fit and transform in one step.
     */
    public float[][] fitTransform(float[][] X) {
        fit(X);
        return transform(X);
    }

    /**
     * Transform a single sample (for online prediction).
     */
    public float[] transformSingle(float[] x) {
        float[][] single = new float[][] { x };
        return transform(single)[0];
    }

    /**
     * Inverse transform (if possible).
     */
    public float[][] inverseTransform(float[][] X) {
        throw new UnsupportedOperationException(
                this.getClass().getSimpleName() + " does not implement inverse_transform");
    }

    /**
     * Get feature names after transformation.
     */
    public List<String> getFeatureNames(List<String> inputFeatures) {
        // Default implementation returns input features
        return new ArrayList<>(inputFeatures);
    }

    /**
     * Get number of features produced by this transformer.
     */
    public int getNFeaturesOut() {
        return -1; // Default -1 means unknown or same as input
    }


    @Override
    public void fit(float[][] X, int[] y) {
        // Most transformers don't use labels, but implement this for compatibility
        fit(X);
    }

    @Override
    public int[] predict(float[][] X) {
        throw new UnsupportedOperationException("Transformers are for transformations, not predictions");
    }
}