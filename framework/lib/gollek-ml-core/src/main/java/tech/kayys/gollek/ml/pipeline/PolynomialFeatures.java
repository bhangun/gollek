package tech.kayys.gollek.ml.pipeline;

import tech.kayys.gollek.ml.base.BaseTransformer;
import java.util.*;

/**
 * Generate polynomial and interaction features.
 */
public class PolynomialFeatures extends BaseTransformer {
    private final int degree;
    private final boolean interactionOnly;
    private final boolean includeBias;

    public PolynomialFeatures(int degree) {
        this(degree, false, true);
    }

    public PolynomialFeatures(int degree, boolean interactionOnly, boolean includeBias) {
        this.degree = degree;
        this.interactionOnly = interactionOnly;
        this.includeBias = includeBias;
    }

    @Override
    public void fit(float[][] X) {
        // No fitting needed for polynomial features
        setFitted(true);
    }

    @Override
    public float[][] transform(float[][] X) {
        validateInput(X);
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        List<int[]> combinations = getCombinations(nFeatures, degree, interactionOnly);
        int nOutputFeatures = combinations.size() + (includeBias ? 1 : 0);
        
        float[][] transformed = new float[nSamples][nOutputFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            int col = 0;
            if (includeBias) {
                transformed[i][col++] = 1.0f;
            }
            
            for (int[] combo : combinations) {
                float val = 1.0f;
                for (int idx : combo) {
                    val *= X[i][idx];
                }
                transformed[i][col++] = val;
            }
        }
        
        return transformed;
    }

    private List<int[]> getCombinations(int nFeatures, int degree, boolean interactionOnly) {
        List<int[]> result = new ArrayList<>();
        for (int d = 1; d <= degree; d++) {
            generateCombinations(new int[d], 0, 0, nFeatures, interactionOnly, result);
        }
        return result;
    }

    private void generateCombinations(int[] current, int pos, int start, int nFeatures, 
                                     boolean interactionOnly, List<int[]> result) {
        if (pos == current.length) {
            result.add(current.clone());
            return;
        }

        for (int i = start; i < nFeatures; i++) {
            current[pos] = i;
            generateCombinations(current, pos + 1, interactionOnly ? i + 1 : i, nFeatures, interactionOnly, result);
        }
    }
}
