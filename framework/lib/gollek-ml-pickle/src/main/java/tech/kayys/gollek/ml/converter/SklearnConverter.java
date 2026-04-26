package tech.kayys.gollek.ml.converter;

import tech.kayys.gollek.ml.base.*;
import tech.kayys.gollek.ml.ensemble.*;
import tech.kayys.gollek.ml.tree.*;
import tech.kayys.gollek.ml.linear_model.LinearModel;
import tech.kayys.gollek.ml.svm.SVC;
import tech.kayys.gollek.ml.pickle.*;

import java.util.*;

/**
 * Convert scikit-learn models to Gollek models.
 */
public class SklearnConverter {

    /**
     * Convert pickle-loaded object to Gollek estimator.
     */
    public static BaseEstimator convert(PickleParser.PickleObject pickleObj) {
        String className = pickleObj.getType().getName();

        switch (className) {
            case "RandomForestClassifier":
                return convertRandomForest(pickleObj);
            case "DecisionTreeClassifier":
                return convertDecisionTree(pickleObj);
            case "GradientBoostingClassifier":
                return convertGradientBoosting(pickleObj);
            case "LogisticRegression":
                return convertLogisticRegression(pickleObj);
            case "SVC":
                return convertSVM(pickleObj);
            case "PCA":
                return convertPCA(pickleObj);
            case "StandardScaler":
                return convertStandardScaler(pickleObj);
            default:
                throw new UnsupportedOperationException("Unsupported sklearn class: " + className);
        }
    }

    /**
     * Convert RandomForestClassifier.
     */
    private static RandomForestClassifier convertRandomForest(PickleParser.PickleObject rf) {
        @SuppressWarnings("unchecked")
        List<PickleParser.PickleObject> estimators = rf.getState("estimators_");
        @SuppressWarnings("unchecked")
        Map<String, Object> baseEstimator = rf.getState("base_estimator_");

        int nEstimators = estimators != null ? estimators.size() : (Integer) rf.getState("n_estimators");
        int maxDepth = (Integer) rf.getState("max_depth");
        int minSamplesSplit = (Integer) rf.getState("min_samples_split");

        RandomForestClassifier model = new RandomForestClassifier(
                nEstimators, maxDepth, minSamplesSplit, "gini", "sqrt", true, -1, false);

        // Convert individual trees
        // This would require extracting tree structures from sklearn

        return model;
    }

    /**
     * Convert DecisionTreeClassifier.
     */
    private static DecisionTreeClassifier convertDecisionTree(PickleParser.PickleObject tree) {
        int maxDepth = tree.getState("max_depth") != null ? (Integer) tree.getState("max_depth") : 5;
        int minSamplesSplit = tree.getState("min_samples_split") != null ? (Integer) tree.getState("min_samples_split")
                : 2;

        DecisionTreeClassifier model = new DecisionTreeClassifier(
                maxDepth, minSamplesSplit, 1, "gini", "sqrt");

        // Extract tree structure
        Map<String, Object> treeState = (Map<String, Object>) tree.getState("tree_");
        int[] childrenLeft = (int[]) treeState.get("children_left");
        int[] childrenRight = (int[]) treeState.get("children_right");
        int[] feature = (int[]) treeState.get("feature");
        double[] threshold = (double[]) treeState.get("threshold");
        double[] value = (double[]) treeState.get("value");

        model.setRoot(buildClassifierNode(0, childrenLeft, childrenRight, feature, threshold, value));
        return model;
    }

    private static DecisionTreeClassifier.Node buildClassifierNode(int nodeIdx, int[] left, int[] right, int[] feature, double[] threshold, double[] value) {
        if (left[nodeIdx] == -1) {
            // Leaf node. value[nodeIdx] is usually double[1][n_classes] flattened or similar.
            // Simplified: extract the class with max count.
            // In practice, sklearn value is double[node_count][1][n_classes]
            int nClasses = value.length / left.length;
            double[] probs = new double[nClasses];
            int maxClass = 0;
            for (int i = 0; i < nClasses; i++) {
                probs[i] = value[nodeIdx * nClasses + i];
                if (probs[i] > probs[maxClass]) maxClass = i;
            }
            // Normalize probs
            double sum = Arrays.stream(probs).sum();
            if (sum > 0) for (int i = 0; i < nClasses; i++) probs[i] /= sum;
            
            return new DecisionTreeClassifier.LeafNode(maxClass, probs);
        } else {
            return new DecisionTreeClassifier.SplitNode(
                feature[nodeIdx],
                threshold[nodeIdx],
                buildClassifierNode(left[nodeIdx], left, right, feature, threshold, value),
                buildClassifierNode(right[nodeIdx], left, right, feature, threshold, value),
                0.0 // Gain not easily available from pickle
            );
        }
    }

    /**
     * Convert GradientBoostingClassifier.
     */
    private static GradientBoostingClassifier convertGradientBoosting(PickleParser.PickleObject gb) {
        int nEstimators = (Integer) gb.getState("n_estimators");
        double learningRate = (Double) gb.getState("learning_rate");
        int maxDepth = (Integer) gb.getState("max_depth");
        int minSamplesSplit = (Integer) gb.getState("min_samples_split");
        String loss = (String) gb.getState("loss");

        GradientBoostingClassifier model = new GradientBoostingClassifier(
                nEstimators, learningRate, maxDepth, minSamplesSplit, loss, 1.0, 10);

        // Extract estimators
        @SuppressWarnings("unchecked")
        List<PickleParser.PickleObject[]> estimators = (List<PickleParser.PickleObject[]>) gb.getState("estimators_");
        
        // GBC in sklearn has estimators_ as a list of arrays (one array per estimator, each array contains regressors for each class)
        // For binary classification, it's usually 1 regressor per stage.
        
        for (PickleParser.PickleObject[] stage : estimators) {
            for (PickleParser.PickleObject treeObj : stage) {
                DecisionTreeRegressor regressor = convertDecisionTreeRegressor(treeObj);
                model.addTree(regressor, learningRate);
            }
        }

        return model;
    }

    private static DecisionTreeRegressor convertDecisionTreeRegressor(PickleParser.PickleObject tree) {
        Map<String, Object> treeState = (Map<String, Object>) tree.getState("tree_");
        int[] childrenLeft = (int[]) treeState.get("children_left");
        int[] childrenRight = (int[]) treeState.get("children_right");
        int[] feature = (int[]) treeState.get("feature");
        double[] threshold = (double[]) treeState.get("threshold");
        double[] value = (double[]) treeState.get("value");

        DecisionTreeRegressor model = new DecisionTreeRegressor();
        model.setRoot(buildRegressorNode(0, childrenLeft, childrenRight, feature, threshold, value));
        return model;
    }

    private static DecisionTreeRegressor.Node buildRegressorNode(int nodeIdx, int[] left, int[] right, int[] feature, double[] threshold, double[] value) {
        if (left[nodeIdx] == -1) {
            return new DecisionTreeRegressor.Node(value[nodeIdx]);
        } else {
            DecisionTreeRegressor.Node node = new DecisionTreeRegressor.Node(feature[nodeIdx], (float) threshold[nodeIdx]);
            node.left = buildRegressorNode(left[nodeIdx], left, right, feature, threshold, value);
            node.right = buildRegressorNode(right[nodeIdx], left, right, feature, threshold, value);
            return node;
        }
    }

    /**
     * Convert LogisticRegression.
     */
    private static LinearModel convertLogisticRegression(PickleParser.PickleObject lr) {
        double[] coef = (double[]) lr.getState("coef_");
        double intercept = (Double) lr.getState("intercept_");
        double C = (Double) lr.getState("C");
        String penalty = (String) lr.getState("penalty");

        String gollekPenalty = "l2".equals(penalty) ? "l2" : "none";
        double alpha = 1.0 / C;

        LinearModel model = new LinearModel(gollekPenalty, alpha, 1.0, 1e-4, 1000, 0.01);

        // Set coefficients
        model.setCoefficients(coef);
        model.setIntercept(intercept);

        return model;
    }

    /**
     * Convert Support Vector Machine.
     */
    private static SVC convertSVM(PickleParser.PickleObject svm) {
        double C = (Double) svm.getState("C");
        String kernel = (String) svm.getState("kernel");
        double gamma = svm.getState("gamma") != null ? (Double) svm.getState("gamma") : 0.0;
        double tol = (Double) svm.getState("tol");
        int maxIter = (Integer) svm.getState("max_iter");

        SVC model = new SVC(C, kernel, String.valueOf(gamma), 3, 0.0, tol, maxIter);

        // Extract support vectors
        double[][] supportVectors = (double[][]) svm.getState("support_vectors_");
        double[] dualCoef = (double[]) svm.getState("dual_coef_");
        double intercept = (Double) svm.getState("intercept_");
        
        // Convert support vectors to float[][]
        float[][] svFloat = new float[supportVectors.length][];
        for (int i = 0; i < supportVectors.length; i++) {
            svFloat[i] = new float[supportVectors[i].length];
            for (int j = 0; j < supportVectors[i].length; j++) {
                svFloat[i][j] = (float) supportVectors[i][j];
            }
        }
        
        // sklearn's dual_coef_ is (n_classes-1, n_SV)
        // For binary, it's (1, n_SV)
        // supportVectorLabels should be +1/-1 based on sklearn's decision function
        // This is a simplification.
        int[] labels = new int[dualCoef.length];
        for (int i = 0; i < labels.length; i++) labels[i] = dualCoef[i] > 0 ? 1 : -1;

        model.setSupportVectors(svFloat, labels, dualCoef, -intercept);

        return model;
    }

    /**
     * Convert PCA.
     */
    private static tech.kayys.gollek.ml.pipeline.PCA convertPCA(PickleParser.PickleObject pca) {
        int nComponents = (Integer) pca.getState("n_components_");
        double[] mean = (double[]) pca.getState("mean_");
        double[] explainedVariance = (double[]) pca.getState("explained_variance_");
        double[][] components = (double[][]) pca.getState("components_");

        tech.kayys.gollek.ml.pipeline.PCA model = new tech.kayys.gollek.ml.pipeline.PCA(nComponents);

        model.setParameters(components, mean, explainedVariance);

        return model;
    }

    /**
     * Convert StandardScaler.
     */
    private static tech.kayys.gollek.ml.pipeline.StandardScaler convertStandardScaler(
            PickleParser.PickleObject scaler) {
        double[] mean = (double[]) scaler.getState("mean_");
        double[] scale = (double[]) scaler.getState("scale_");

        tech.kayys.gollek.ml.pipeline.StandardScaler model = new tech.kayys.gollek.ml.pipeline.StandardScaler();

        model.setParameters(mean, scale);

        return model;
    }

    /**
     * Extract numpy array from pickle state.
     */
    private static float[] extractFloatArray(Object obj) {
        if (obj instanceof byte[]) {
            byte[] bytes = (byte[]) obj;
            float[] result = new float[bytes.length / 4];
            for (int i = 0; i < result.length; i++) {
                result[i] = Float.intBitsToFloat(
                        ((bytes[i * 4] & 0xFF) << 0) |
                                ((bytes[i * 4 + 1] & 0xFF) << 8) |
                                ((bytes[i * 4 + 2] & 0xFF) << 16) |
                                ((bytes[i * 4 + 3] & 0xFF) << 24));
            }
            return result;
        } else if (obj instanceof double[]) {
            double[] doubles = (double[]) obj;
            float[] result = new float[doubles.length];
            for (int i = 0; i < doubles.length; i++) {
                result[i] = (float) doubles[i];
            }
            return result;
        }
        return null;
    }
}