package tech.kayys.gollek.ml.base;

import io.quarkus.runtime.annotations.RegisterForReflection;
import java.util.*;
import java.io.*;
import java.lang.reflect.*;

/**
 * Base class for all estimators in Gollek ML.
 * Provides common functionality: parameter validation, cloning, and
 * get_params/set_params.
 */
@RegisterForReflection(registerFullHierarchy = true)
public abstract class BaseEstimator implements Cloneable, Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Get parameters for this estimator.
     * Used for grid search and model serialization.
     */
    public Map<String, Object> getParams() {
        Map<String, Object> params = new LinkedHashMap<>();

        // Use reflection to get all public fields and methods
        for (Field field : getClass().getDeclaredFields()) {
            if (isParameterField(field)) {
                field.setAccessible(true);
                try {
                    params.put(field.getName(), field.get(this));
                } catch (IllegalAccessException e) {
                    // Skip inaccessible fields
                }
            }
        }

        return params;
    }

    /**
     * Set parameters for this estimator.
     * Chainable for convenient construction.
     */
    public BaseEstimator setParams(Map<String, Object> params) {
        for (Map.Entry<String, Object> entry : params.entrySet()) {
            try {
                Field field = getClass().getDeclaredField(entry.getKey());
                field.setAccessible(true);
                field.set(this, entry.getValue());
            } catch (NoSuchFieldException | IllegalAccessException e) {
                throw new IllegalArgumentException("Invalid parameter: " + entry.getKey(), e);
            }
        }
        return this;
    }

    /**
     * Create a deep copy of the estimator.
     */
    public BaseEstimator clone() {
        try {
            BaseEstimator clone = (BaseEstimator) super.clone();
            // Deep clone internal state
            for (Field field : getClass().getDeclaredFields()) {
                field.setAccessible(true);
                Object value;
                try {
                    value = field.get(this);
                } catch (IllegalAccessException e) {
                    continue;
                }
                if (value instanceof Cloneable) {
                    try {
                        Method cloneMethod = value.getClass().getMethod("clone");
                        field.set(clone, cloneMethod.invoke(value));
                    } catch (Exception e) {
                        // Keep reference if clone not available
                    }
                }
            }
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Failed to clone estimator", e);
        }
    }

    private boolean isFitted = false;

    /**
     * Check if the estimator has been fitted.
     */
    public boolean isFitted() {
        return isFitted;
    }

    protected void setFitted(boolean fitted) {
        this.isFitted = fitted;
    }

    /**
     * Validate input data dimensionality.
     */
    protected void validateData(float[][] X, int[] y) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
        if (y == null || y.length != X.length) {
            throw new IllegalArgumentException("Labels must match number of samples");
        }
    }

    protected void validateData(float[][] X, int[][] y) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
        if (y == null || y.length != X.length) {
            throw new IllegalArgumentException("Labels must match number of samples");
        }
    }

    /**
     * Validate input data for prediction.
     */
    protected void validateInput(float[][] X) {
        if (!isFitted()) {
            throw new IllegalStateException("Estimator must be fitted before prediction");
        }
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
    }

    private boolean isParameterField(Field field) {
        int modifiers = field.getModifiers();
        return !Modifier.isStatic(modifiers) &&
                !Modifier.isFinal(modifiers) &&
                !Modifier.isTransient(modifiers);
    }

    // Abstract methods that all estimators must implement
    public abstract void fit(float[][] X, int[] y);

    public void fit(float[][] X, int[][] y) {
        throw new UnsupportedOperationException("This estimator does not support multi-output training");
    }

    public abstract int[] predict(float[][] X);

    public int predictSingle(float[] x) {
        return (int) Math.round(predictSingleValue(x));
    }

    public double[] predictValues(float[][] X) {
        int[] labels = predict(X);
        double[] values = new double[labels.length];
        for (int i = 0; i < labels.length; i++) values[i] = labels[i];
        return values;
    }

    public double predictSingleValue(float[] x) {
        float[][] X = new float[][] { x };
        return predictValues(X)[0];
    }

    // Optional methods with default implementations
    public double[][] predictProba(float[][] X) {
        throw new UnsupportedOperationException("This estimator does not support probability predictions");
    }

    public double score(float[][] X, int[] y) {
        int[] predictions = predict(X);
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == y[i])
                correct++;
        }
        return (double) correct / predictions.length;
    }
}