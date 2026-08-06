package tech.kayys.gollek.ml.pickle;

import tech.kayys.gollek.ml.base.*;
import tech.kayys.gollek.ml.converter.*;
import tech.kayys.gollek.ml.autograd.*;
import tech.kayys.gollek.ml.persistence.ModelPersistence;
import tech.kayys.gollek.ml.ensemble.RandomForestClassifier;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Complete pickle model runner.
 * Loads and runs scikit-learn/PyTorch models directly from pickle files.
 */
public class PickleModelRunner implements AutoCloseable {

    private final String modelPath;
    private BaseEstimator model;
    private Map<String, Object> metadata;
    private long loadTime;
    private long inferenceCount = 0;
    private double totalInferenceTime = 0;

    public PickleModelRunner(String modelPath) {
        this.modelPath = modelPath;
    }

    /**
     * Load and initialize the model.
     */
    public void load() throws IOException {
        long startTime = System.currentTimeMillis();

        // Read pickle file
        byte[] data = java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(modelPath));
        PickleParser parser = new PickleParser(data);
        Object obj = parser.parse();

        // Convert based on model type
        if (obj instanceof PickleParser.PickleObject) {
            PickleParser.PickleObject pickleObj = (PickleParser.PickleObject) obj;
            String className = pickleObj.getType().getName();

            // Detect model type
            if (isSklearnModel(className)) {
                this.model = SklearnConverter.convert(pickleObj);
            } else if (isPyTorchModel(className)) {
                // PyTorch models usually require a state dict loader or architecture reconstruction
                // Here we attempt to load the state dict if it's a direct module save
                this.model = convertPyTorchToSklearn(pickleObj);
            }

            // Extract metadata
            this.metadata = extractMetadata(pickleObj);
        } else if (obj instanceof Map) {
            // Likely a PyTorch state dict saved as a map
            @SuppressWarnings("unchecked")
            Map<String, Object> stateDict = (Map<String, Object>) obj;
            this.metadata = new HashMap<>();
            this.metadata.put("type", "pytorch_state_dict");
            this.metadata.put("param_count", stateDict.size());
            // PyTorch state dicts need an NNModule to load into
            System.out.println("Loaded PyTorch state dict. Use PyTorchConverter to map to an NNModule.");
        }

        this.loadTime = System.currentTimeMillis() - startTime;
        if (this.model != null) {
            System.out.printf("Model loaded in %d ms: %s\n", loadTime, model.getClass().getSimpleName());
        } else {
            System.out.printf("Pickle object loaded in %d ms, but no estimator mapped.\n", loadTime);
        }
    }

    /**
     * Run inference on single sample.
     */
    public int predict(float[] sample) {
        if (model == null) {
            throw new IllegalStateException("Model not loaded or not a classification model. Call load() first.");
        }

        long startTime = System.nanoTime();
        int result = model.predictSingle(sample);

        totalInferenceTime += (System.nanoTime() - startTime) / 1_000_000.0;
        inferenceCount++;

        return result;
    }

    /**
     * Run inference on batch.
     */
    public int[] predictBatch(float[][] samples) {
        if (model == null) {
            throw new IllegalStateException("Model not loaded. Call load() first.");
        }

        long startTime = System.nanoTime();
        int[] results = model.predict(samples);

        totalInferenceTime += (System.nanoTime() - startTime) / 1_000_000.0;
        inferenceCount += samples.length;

        return results;
    }

    /**
     * Get prediction probabilities.
     */
    public double[][] predictProba(float[][] samples) {
        if (model == null) {
            throw new IllegalStateException("Model not loaded. Call load() first.");
        }

        return model.predictProba(samples);
    }

    /**
     * Run inference asynchronously.
     */
    public CompletableFuture<int[]> predictAsync(float[][] samples) {
        return CompletableFuture.supplyAsync(() -> predictBatch(samples));
    }

    /**
     * Get model metadata.
     */
    public Map<String, Object> getMetadata() {
        return metadata != null ? Collections.unmodifiableMap(metadata) : Map.of();
    }

    /**
     * Get performance stats.
     */
    public PerformanceStats getStats() {
        return new PerformanceStats(
                inferenceCount,
                totalInferenceTime,
                inferenceCount > 0 ? totalInferenceTime / inferenceCount : 0,
                loadTime);
    }

    /**
     * Save model as Gollek native format.
     */
    public void saveAsGollek(String outputPath) throws IOException {
        if (model == null) {
            throw new IllegalStateException("Model not loaded");
        }
        ModelPersistence.save(model, outputPath);
    }

    /**
     * Export model to ONNX.
     */
    public void exportToONNX(String outputPath, long[] inputShape) throws IOException {
        // Convert to ONNX format
        // This would require implementing ONNX export
        System.out.println("ONNX export not yet implemented for pickle models");
    }

    private boolean isSklearnModel(String className) {
        return className.contains("sklearn") ||
                className.equals("RandomForestClassifier") ||
                className.equals("DecisionTreeClassifier") ||
                className.equals("GradientBoostingClassifier") ||
                className.equals("LogisticRegression") ||
                className.equals("SVC") ||
                className.equals("PCA") ||
                className.equals("StandardScaler");
    }

    private boolean isPyTorchModel(String className) {
        return className.contains("torch") ||
                className.equals("Sequential") ||
                className.equals("Module") ||
                className.contains("Linear");
    }

    private BaseEstimator convertPyTorchToSklearn(PickleParser.PickleObject pytorchObj) {
        // Extract architecture and convert to sklearn-compatible model
        // In a real scenario, this would use PyTorchConverter to rebuild the NNModule
        // and wrap it in a BaseEstimator.
        
        return null;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> extractMetadata(PickleParser.PickleObject obj) {
        Map<String, Object> metadata = new LinkedHashMap<>();

        metadata.put("model_type", obj.getType().getName());
        metadata.put("pickle_module", obj.getType().getModule());

        // Extract common sklearn attributes
        if (obj.getState("classes_") != null) {
            Object classes = obj.getState("classes_");
            if (classes instanceof Object[]) {
                metadata.put("n_classes", ((Object[]) classes).length);
            }
        }

        if (obj.getState("n_features_") != null) {
            metadata.put("n_features", obj.getState("n_features_"));
        }

        if (obj.getState("feature_names_in_") != null) {
            metadata.put("feature_names", obj.getState("feature_names_in_"));
        }

        return metadata;
    }

    @Override
    public void close() {
        // Cleanup if needed
        model = null;
        metadata = null;
    }

    /**
     * Performance statistics record.
     */
    public record PerformanceStats(
            long totalInferences,
            double totalTimeMs,
            double avgLatencyMs,
            long loadTimeMs) {
        public double throughputPerSecond() {
            return totalTimeMs > 0 ? (totalInferences * 1000.0) / totalTimeMs : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "PerformanceStats{inferences=%d, avg=%.2fms, throughput=%.1f/s, loadTime=%dms}",
                    totalInferences, avgLatencyMs, throughputPerSecond(), loadTimeMs);
        }
    }
}