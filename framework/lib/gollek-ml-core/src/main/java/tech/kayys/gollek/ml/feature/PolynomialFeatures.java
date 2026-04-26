package tech.kayys.gollek.ml.feature;

import java.util.*;
import tech.kayys.gollek.ml.base.BaseTransformer;

/**
 * Polynomial feature expansion.
 */
public class PolynomialFeatures extends BaseTransformer {
    private final int degree;
    private final boolean includeBias;
    private final boolean interactionOnly;

    private int nFeaturesOut;
    private List<int[]> combinations;

    public PolynomialFeatures(int degree) {
        this(degree, true, false);
    }

    public PolynomialFeatures(int degree, boolean includeBias, boolean interactionOnly) {
        this.degree = degree;
        this.includeBias = includeBias;
        this.interactionOnly = interactionOnly;
    }

    @Override
    public void fit(float[][] X) {
        int nFeatures = X[0].length;
        combinations = new ArrayList<>();

        // Generate all polynomial combinations
        if (includeBias) {
            combinations.add(new int[0]); // bias term
        }

        for (int d = 1; d <= degree; d++) {
            generateCombinations(new int[d], 0, nFeatures, 0, d);
        }

        nFeaturesOut = combinations.size();
    }

    private void generateCombinations(int[] combo, int pos, int nFeatures,
            int start, int degree) {
        if (pos == degree) {
            // Check if interaction only (no powers > 1)
            if (interactionOnly) {
                Set<Integer> unique = new HashSet<>();
                for (int v : combo)
                    unique.add(v);
                if (unique.size() != combo.length)
                    return; // Skip if any feature repeated
            }
            combinations.add(combo.clone());
            return;
        }

        for (int i = start; i < nFeatures; i++) {
            combo[pos] = i;
            generateCombinations(combo, pos + 1, nFeatures, i, degree);
        }
    }

    @Override
    public float[][] transform(float[][] X) {
        int nSamples = X.length;
        float[][] transformed = new float[nSamples][nFeaturesOut];

        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < combinations.size(); j++) {
                int[] combo = combinations.get(j);
                double value = 1.0;
                for (int idx : combo) {
                    value *= X[i][idx];
                }
                transformed[i][j] = (float) value;
            }
        }

        return transformed;
    }

    public int getNFeaturesOut() {
        return nFeaturesOut;
    }
}
