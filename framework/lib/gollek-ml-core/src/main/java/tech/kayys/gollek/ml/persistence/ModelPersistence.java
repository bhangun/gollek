package tech.kayys.gollek.ml.persistence;

import java.io.*;
import java.util.*;
import java.util.zip.*;
import tech.kayys.gollek.ml.base.*;
import tech.kayys.gollek.ml.tree.*;
import tech.kayys.gollek.ml.ensemble.*;
import tech.kayys.gollek.ml.linear_model.*;

/**
 * Save and load scikit-learn compatible models.
 * Supports all estimators and transformers.
 */
public class ModelPersistence {

    /**
     * Save model to file with optional compression.
     */
    public static void save(BaseEstimator model, String path) throws IOException {
        save(model, new File(path));
    }

    public static void save(BaseEstimator model, File file) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new GZIPOutputStream(new FileOutputStream(file)))) {

            // Extract model state
            Map<String, Object> state = extractState(model);
            oos.writeObject(state);
        }
    }

    /**
     * Load model from file.
     */
    @SuppressWarnings("unchecked")
    public static <T extends BaseEstimator> T load(String path) throws IOException, ClassNotFoundException {
        return load(new File(path));
    }

    @SuppressWarnings("unchecked")
    public static <T extends BaseEstimator> T load(File file) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(
                new GZIPInputStream(new FileInputStream(file)))) {

            Map<String, Object> state = (Map<String, Object>) ois.readObject();
            String className = (String) state.get("_class");

            // Instantiate model
            Class<?> modelClass = Class.forName(className);
            T model = (T) modelClass.getDeclaredConstructor().newInstance();

            // Restore state
            restoreState(model, state);

            return model;
        } catch (Exception e) {
            throw new RuntimeException("Failed to load model", e);
        }
    }

    /**
     * Export model to PMML for cross-platform deployment.
     */
    public static String toPMML(BaseEstimator model, String modelName,
            List<String> featureNames, List<String> targetNames) {
        StringBuilder pmml = new StringBuilder();

        pmml.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        pmml.append("<PMML version=\"4.4\" xmlns=\"http://www.dmg.org/PMML-4_4\">\n");
        pmml.append("  <Header>\n");
        pmml.append("    <Application name=\"Gollek ML\"/>\n");
        pmml.append("    <Timestamp>").append(new Date()).append("</Timestamp>\n");
        pmml.append("  </Header>\n");
        pmml.append("  <DataDictionary>\n");

        // Define data dictionary
        for (String feature : featureNames) {
            pmml.append("    <DataField name=\"").append(feature)
                    .append("\" optype=\"continuous\" dataType=\"double\"/>\n");
        }
        for (String target : targetNames) {
            pmml.append("    <DataField name=\"").append(target)
                    .append("\" optype=\"categorical\" dataType=\"string\"/>\n");
        }
        pmml.append("  </DataDictionary>\n");

        // Add model-specific XML
        if (model instanceof DecisionTreeClassifier) {
            pmml.append(exportTreeToPMML((DecisionTreeClassifier) model, featureNames));
        } else if (model instanceof RandomForestClassifier) {
            pmml.append(exportForestToPMML((RandomForestClassifier) model, featureNames));
        } else if (model instanceof LinearModel) {
            pmml.append(exportLinearToPMML((LinearModel) model, featureNames, targetNames));
        }

        pmml.append("</PMML>");
        return pmml.toString();
    }

    private static String exportTreeToPMML(DecisionTreeClassifier tree, List<String> features) {
        // Simplified - would need to traverse the tree structure
        return "    <TreeModel modelName=\"DecisionTree\" functionName=\"classification\">\n" +
                "      <Node id=\"1\" score=\"0\">\n" +
                "        <True/>\n" +
                "      </Node>\n" +
                "    </TreeModel>\n";
    }

    private static String exportForestToPMML(RandomForestClassifier forest, List<String> features) {
        return "    <MiningModel modelName=\"RandomForest\" functionName=\"classification\">\n" +
                "      <MiningSchema>\n" +
                "        <MiningField name=\"predicted\"/>\n" +
                "      </MiningSchema>\n" +
                "      <Segmentation multipleModelMethod=\"majorityVote\">\n" +
                "        <!-- Segment for each tree -->\n" +
                "      </Segmentation>\n" +
                "    </MiningModel>\n";
    }

    private static String exportLinearToPMML(LinearModel model, List<String> features, List<String> targets) {
        double[] coef = model.getCoefficients();
        double intercept = model.getIntercept();

        StringBuilder pmml = new StringBuilder();
        pmml.append("    <RegressionModel modelName=\"LinearRegression\" functionName=\"regression\">\n");
        pmml.append("      <MiningSchema>\n");
        pmml.append("        <MiningField name=\"predicted\"/>\n");
        pmml.append("      </MiningSchema>\n");
        pmml.append("      <RegressionTable intercept=\"").append(intercept).append("\">\n");

        for (int i = 0; i < features.size() && i < coef.length; i++) {
            pmml.append("        <NumericPredictor name=\"").append(features.get(i))
                    .append("\" coefficient=\"").append(coef[i]).append("\"/>\n");
        }

        pmml.append("      </RegressionTable>\n");
        pmml.append("    </RegressionModel>\n");
        return pmml.toString();
    }

    /**
     * Extract state from estimator for serialization.
     */
    private static Map<String, Object> extractState(BaseEstimator model) {
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("_class", model.getClass().getName());

        // Use reflection to extract fields
        try {
            for (java.lang.reflect.Field field : model.getClass().getDeclaredFields()) {
                field.setAccessible(true);
                Object value = field.get(model);
                if (value != null && isSerializable(value)) {
                    state.put(field.getName(), value);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to extract model state", e);
        }

        return state;
    }

    /**
     * Restore state to estimator from serialized data.
     */
    private static void restoreState(BaseEstimator model, Map<String, Object> state) {
        try {
            for (java.lang.reflect.Field field : model.getClass().getDeclaredFields()) {
                field.setAccessible(true);
                if (state.containsKey(field.getName())) {
                    field.set(model, state.get(field.getName()));
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to restore model state", e);
        }
    }

    private static boolean isSerializable(Object obj) {
        return obj instanceof Serializable ||
                obj instanceof Number ||
                obj instanceof String ||
                obj instanceof Boolean ||
                obj instanceof Date ||
                (obj instanceof Object[] && ((Object[]) obj).length > 0 &&
                        isSerializable(((Object[]) obj)[0]));
    }
}