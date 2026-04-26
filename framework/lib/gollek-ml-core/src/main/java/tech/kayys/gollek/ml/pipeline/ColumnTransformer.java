package tech.kayys.gollek.ml.pipeline;

import tech.kayys.gollek.ml.base.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * ColumnTransformer applies different transformers to different columns.
 * Essential for heterogeneous datasets with mixed data types.
 */
public class ColumnTransformer extends BaseTransformer {

    private final List<TransformerColumn> transformers;
    private final String remainder; // "drop", "passthrough"
    private int nFeaturesOut;
    private int nFeaturesIn;

    public ColumnTransformer() {
        this.transformers = new ArrayList<>();
        this.remainder = "drop";
    }

    public ColumnTransformer addTransformer(String name, BaseTransformer transformer, int[] columns) {
        transformers.add(new TransformerColumn(name, transformer, columns));
        return this;
    }

    @Override
    public void fit(float[][] X) {
        validateInput(X);
        this.nFeaturesIn = X[0].length;
        Set<Integer> usedColumns = new HashSet<>();

        // Fit each transformer on its columns
        for (TransformerColumn tc : transformers) {
            float[][] subset = extractColumns(X, tc.columns);
            tc.transformer.fit(subset);
            for (int col : tc.columns) {
                usedColumns.add(col);
            }
        }

        // Calculate output feature count
        nFeaturesOut = 0;
        for (TransformerColumn tc : transformers) {
            float[][] subset = extractColumns(X, tc.columns);
            float[][] transformed = tc.transformer.transform(subset);
            nFeaturesOut += transformed[0].length;
        }
        
        if ("passthrough".equals(remainder)) {
            for (int i = 0; i < nFeaturesIn; i++) {
                if (!usedColumns.contains(i)) nFeaturesOut++;
            }
        }

        setFitted(true);
    }

    @Override
    public float[][] transform(float[][] X) {
        validateInput(X);
        int nSamples = X.length;
        float[][] result = new float[nSamples][nFeaturesOut];
        int colOffset = 0;
        Set<Integer> usedColumns = new HashSet<>();

        for (TransformerColumn tc : transformers) {
            float[][] subset = extractColumns(X, tc.columns);
            float[][] transformed = tc.transformer.transform(subset);
            int nOut = transformed[0].length;
            for (int i = 0; i < nSamples; i++) {
                System.arraycopy(transformed[i], 0, result[i], colOffset, nOut);
            }
            colOffset += nOut;
            for (int col : tc.columns) usedColumns.add(col);
        }

        if ("passthrough".equals(remainder)) {
            for (int j = 0; j < nFeaturesIn; j++) {
                if (!usedColumns.contains(j)) {
                    for (int i = 0; i < nSamples; i++) {
                        result[i][colOffset] = X[i][j];
                    }
                    colOffset++;
                }
            }
        }

        return result;
    }

    private float[][] extractColumns(float[][] X, int[] columns) {
        int nSamples = X.length;
        float[][] subset = new float[nSamples][columns.length];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < columns.length; j++) {
                subset[i][j] = X[i][columns[j]];
            }
        }
        return subset;
    }

    @Override
    public List<String> getFeatureNames(List<String> inputFeatures) {
        List<String> names = new ArrayList<>();
        Set<Integer> usedColumns = new HashSet<>();
        for (TransformerColumn tc : transformers) {
            List<String> tcInputNames = new ArrayList<>();
            for (int col : tc.columns) {
                tcInputNames.add(inputFeatures.get(col));
                usedColumns.add(col);
            }
            names.addAll(tc.transformer.getFeatureNames(tcInputNames));
        }
        if ("passthrough".equals(remainder)) {
            for (int i = 0; i < nFeaturesIn; i++) {
                if (!usedColumns.contains(i)) names.add(inputFeatures.get(i));
            }
        }
        return names;
    }

    @Override
    public int getNFeaturesOut() {
        return nFeaturesOut;
    }

    private static class TransformerColumn {
        final String name;
        final BaseTransformer transformer;
        final int[] columns;

        TransformerColumn(String name, BaseTransformer transformer, int[] columns) {
            this.name = name;
            this.transformer = transformer;
            this.columns = columns;
        }
    }
}