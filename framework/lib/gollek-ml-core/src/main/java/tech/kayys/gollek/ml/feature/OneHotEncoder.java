package tech.kayys.gollek.ml.feature;

import tech.kayys.gollek.ml.base.BaseTransformer;
import java.util.*;

/**
 * One-Hot Encoder for categorical features.
 */
public class OneHotEncoder extends BaseTransformer {
    private final boolean sparseOutput;
    private final int handleUnknown; // 0=error, 1=ignore
    private Map<Integer, Map<Float, Integer>> categoryMaps;
    private int[] nValues;
    private int[] featureIndices;

    public OneHotEncoder() {
        this(false, 0, null);
    }

    public OneHotEncoder(boolean sparseOutput, int handleUnknown, int[] featureIndices) {
        this.sparseOutput = sparseOutput;
        this.handleUnknown = handleUnknown;
        this.featureIndices = featureIndices;
    }

    @Override
    public void fit(float[][] X) {
        if (featureIndices == null) {
            featureIndices = new int[X[0].length];
            for (int i = 0; i < featureIndices.length; i++) featureIndices[i] = i;
        }
        
        categoryMaps = new LinkedHashMap<>();
        nValues = new int[featureIndices.length];

        for (int idx = 0; idx < featureIndices.length; idx++) {
            int col = featureIndices[idx];
            Map<Float, Integer> catMap = new LinkedHashMap<>();

            for (float[] row : X) {
                float value = row[col];
                if (!catMap.containsKey(value)) {
                    catMap.put(value, catMap.size());
                }
            }

            categoryMaps.put(col, catMap);
            nValues[idx] = catMap.size();
        }
        setFitted(true);
    }

    @Override
    public float[][] transform(float[][] X) {
        validateInput(X);
        
        // Calculate output dimensions
        int totalFeatures = 0;
        for (int n : nValues)
            totalFeatures += n;

        int nSamples = X.length;
        float[][] transformed = new float[nSamples][totalFeatures];

        int offset = 0;
        for (int idx = 0; idx < featureIndices.length; idx++) {
            int col = featureIndices[idx];
            Map<Float, Integer> catMap = categoryMaps.get(col);

            for (int i = 0; i < nSamples; i++) {
                float value = X[i][col];
                Integer catIdx = catMap.get(value);

                if (catIdx != null) {
                    transformed[i][offset + catIdx] = 1.0f;
                } else if (handleUnknown == 0) {
                    throw new IllegalArgumentException("Unknown category: " + value);
                }
            }

            offset += catMap.size();
        }

        return transformed;
    }

    @Override
    public List<String> getFeatureNames(List<String> inputFeatures) {
        List<String> featureNames = new ArrayList<>();

        for (int idx = 0; idx < featureIndices.length; idx++) {
            int col = featureIndices[idx];
            String inputName = inputFeatures.get(col);
            Map<Float, Integer> catMap = categoryMaps.get(col);

            for (Float category : catMap.keySet()) {
                featureNames.add(inputName + "_" + category);
            }
        }

        return featureNames;
    }
}