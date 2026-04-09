package tech.kayys.gollek.sdk.litert;

import tech.kayys.gollek.sdk.litert.config.LiteRTConfig;
import tech.kayys.gollek.sdk.litert.inference.LiteRTInferenceEngine;
import tech.kayys.gollek.sdk.litert.model.LiteRTModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.io.Closeable;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * LiteRT SDK Manager - High-level API for LiteRT inference.
 * 
 * Provides a unified interface for:
 * - Model loading and management
 * - Inference execution (sync & async)
 * - Batch processing
 * - Model introspection
 * - Performance metrics
 * 
 * Example usage:
 * 
 * <pre>
 * LiteRTSdk sdk = new LiteRTSdk();
 * 
 * // Load a model
 * sdk.loadModel("mobilenet", "/models/mobilenet.litertlm");
 * 
 * // Run inference
 * InferenceResponse response = sdk.infer(
 *         InferenceRequest.builder()
 *                 .model("mobilenet")
 *                 .inputData(inputTensor)
 *                 .build());
 * 
 * // Clean up
 * sdk.close();
 * </pre>
 */
public class LiteRTSdk implements Closeable {

    private final LiteRTConfig config;
    private final LiteRTInferenceEngine inferenceEngine;
    private volatile boolean closed = false;

    /**
     * Create LiteRT SDK with default configuration.
     */
    public LiteRTSdk() {
        this(LiteRTConfig.builder().build());
    }

    /**
     * Create LiteRT SDK with custom configuration.
     * 
     * @param config LiteRT configuration
     */
    public LiteRTSdk(LiteRTConfig config) {
        this.config = config;
        this.inferenceEngine = new LiteRTInferenceEngine(config);
    }

    /**
     * Load a LiteRT model from file path.
     * 
     * @param modelId   Unique identifier for the model
     * @param modelPath Path to the .litertlm model file
     * @throws LiteRTException if model loading fails
     */
    public void loadModel(String modelId, Path modelPath) {
        checkClosed();
        inferenceEngine.loadModel(modelId, modelPath);
    }

    /**
     * Load a LiteRT model from byte array.
     * 
     * @param modelId   Unique identifier for the model
     * @param modelData Model bytes
     * @throws LiteRTException if model loading fails
     */
    public void loadModel(String modelId, byte[] modelData) {
        checkClosed();
        inferenceEngine.loadModel(modelId, modelData);
    }

    /**
     * Run synchronous inference.
     * 
     * @param request Inference request
     * @return Inference response
     * @throws LiteRTException if inference fails
     */
    public InferenceResponse infer(InferenceRequest request) {
        checkClosed();
        return inferenceEngine.infer(request);
    }

    /**
     * Run asynchronous inference.
     * 
     * @param request Inference request
     * @return Future containing inference response
     */
    public CompletableFuture<InferenceResponse> inferAsync(InferenceRequest request) {
        checkClosed();
        return inferenceEngine.inferAsync(request);
    }

    /**
     * Run batch inference.
     * 
     * @param requests List of inference requests
     * @return List of inference responses
     */
    public List<InferenceResponse> inferBatch(List<InferenceRequest> requests) {
        checkClosed();
        return inferenceEngine.inferBatch(requests);
    }

    /**
     * Run batch inference asynchronously.
     * 
     * @param requests List of inference requests
     * @return Future containing list of inference responses
     */
    public CompletableFuture<List<InferenceResponse>> inferBatchAsync(List<InferenceRequest> requests) {
        checkClosed();
        return inferenceEngine.inferBatchAsync(requests);
    }

    /**
     * Get model information.
     * 
     * @param modelId Model identifier
     * @return Model information
     * @throws LiteRTException if model not found
     */
    public LiteRTModelInfo getModelInfo(String modelId) {
        checkClosed();
        return inferenceEngine.getModelInfo(modelId);
    }

    /**
     * List all loaded models.
     * 
     * @return List of model IDs
     */
    public List<String> listModels() {
        checkClosed();
        return inferenceEngine.listModels();
    }

    /**
     * Unload a model.
     * 
     * @param modelId Model identifier
     */
    public void unloadModel(String modelId) {
        checkClosed();
        inferenceEngine.unloadModel(modelId);
    }

    /**
     * Get inference performance metrics.
     * 
     * @param modelId Model identifier (null for all models)
     * @return Metrics map
     */
    public LiteRTMetrics getMetrics(String modelId) {
        checkClosed();
        return inferenceEngine.getMetrics(modelId);
    }

    /**
     * Reset performance metrics.
     */
    public void resetMetrics() {
        checkClosed();
        inferenceEngine.resetMetrics();
    }

    /**
     * Check if LiteRT is available.
     * 
     * @return true if LiteRT is available
     */
    public boolean isAvailable() {
        return inferenceEngine.isAvailable();
    }

    /**
     * Get LiteRT version.
     * 
     * @return Version string
     */
    public String getVersion() {
        return inferenceEngine.getVersion();
    }

    /**
     * Get configuration.
     * 
     * @return LiteRT configuration
     */
    public LiteRTConfig getConfig() {
        return config;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        inferenceEngine.close();
    }

    private void checkClosed() {
        if (closed) {
            throw new IllegalStateException("LiteRTSdk has been closed");
        }
    }
}
