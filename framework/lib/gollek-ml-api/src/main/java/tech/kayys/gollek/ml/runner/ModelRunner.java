package tech.kayys.gollek.ml.runner;

import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.ml.tensor.RunnerDevice;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Unified model runner abstraction that provides a single API for inference
 * across all supported frameworks (LibTorch, LiteRT, ONNX, GGUF, TensorRT, etc.).
 *
 * <p>Instead of writing framework-specific code:
 * <pre>
 *   // Before: framework-specific code
 *   LibTorchRunner torchRunner = new LibTorchRunner(modelPath);
 *   torchRunner.infer(input);
 *
 *   LiteRTRunner litertRunner = new LiteRTRunner(modelPath);
 *   litertRunner.infer(input);
 * </pre>
 *
 * <p>Use the unified API:
 * <pre>
 *   ModelRunner runner = ModelRunner.builder()
 *       .modelPath("model.safetensors")
 *       .format(ModelFormat.SAFETENSORS)
 *       .device(Device.CUDA)
 *       .build();
 *
 *   runner.infer(input);
 * </pre>
 *
 * <h2>Features</h2>
 * <ul>
 *   <li>Framework-agnostic inference - same API for all backends</li>
 *   <li>Automatic backend selection based on model format</li>
 *   <li>Device placement (CPU, CUDA, Metal, ROCm)</li>
 *   <li>Synchronous and asynchronous inference</li>
 *   <li>Batch inference support</li>
 *   <li>Resource management (AutoCloseable)</li>
 * </ul>
 *
 * @see ModelRunnerRegistry
 * @see ModelFormat
 * @since 0.3.0
 */
public interface ModelRunner extends AutoCloseable {

    /**
     * Gets the runner identifier (for logging/monitoring).
     */
    String id();

    /**
     * Gets the underlying framework type.
     */
    ModelFormat format();

    /**
     * Checks if the runner is initialized and ready for inference.
     */
    boolean isReady();

    /**
     * Gets model metadata (input/output shapes, dtypes, etc.).
     */
    ModelMetadata metadata();

    // ── Inference ───────────────────────────────────────────────────────

    /**
     * Runs inference on a single input.
     *
     * @param input model input tensor(s)
     * @return output tensor(s)
     */
    InferenceResult infer(InferenceInput input);

    /**
     * Runs inference on a single input with named inputs.
     *
     * @param inputs named input tensors
     * @return output tensor(s)
     */
    InferenceResult infer(Map<String, InferenceInput> inputs);

    /**
     * Runs inference on a batch of inputs.
     *
     * @param inputs batch of input tensors
     * @return batch of output results
     */
    InferenceResult[] inferBatch(InferenceInput[] inputs);

    /**
     * Runs inference asynchronously.
     *
     * @param input model input tensor(s)
     * @return future with output tensor(s)
     */
    CompletableFuture<InferenceResult> inferAsync(InferenceInput input);

    /**
     * Runs inference asynchronously on a batch.
     *
     * @param inputs batch of input tensors
     * @return future with batch of output results
     */
    CompletableFuture<InferenceResult[]> inferBatchAsync(InferenceInput[] inputs);

    // ── Resource Management ─────────────────────────────────────────────

    /**
     * Gets the device this runner is running on.
     */
    RunnerDevice device();

    /**
     * Gets memory usage statistics.
     */
    MemoryStats memoryStats();

    /**
     * Gets performance metrics.
     */
    PerformanceStats performanceStats();

    /**
     * Resets performance counters.
     */
    void resetStats();

    /**
     * Releases all resources.
     */
    @Override
    void close();

    // ── Builder ─────────────────────────────────────────────────────────

    /**
     * Creates a builder for configuring this runner.
     */
    static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for ModelRunner configuration.
     */
    class Builder {
        private Path modelPath;
        private ModelFormat format;
        private RunnerDevice device = RunnerDevice.AUTO;
        private Map<String, Object> options = Map.of();
        private String id;

        Builder() {}

        public Builder modelPath(Path modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        public Builder modelPath(String modelPath) {
            this.modelPath = Path.of(modelPath);
            return this;
        }

        public Builder format(ModelFormat format) {
            this.format = format;
            return this;
        }

        public Builder device(RunnerDevice device) {
            this.device = device;
            return this;
        }

        public Builder option(String key, Object value) {
            this.options = new java.util.HashMap<>(this.options);
            this.options.put(key, value);
            return this;
        }

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        /**
         * Builds the model runner by auto-selecting the appropriate backend.
         */
        public ModelRunner build() {
            if (modelPath == null) {
                throw new IllegalStateException("modelPath is required");
            }

            // Auto-detect format if not specified
            ModelFormat detectedFormat = format != null ? format : detectFormat(modelPath);
            if (detectedFormat == null) {
                throw new IllegalArgumentException("Cannot detect model format for: " + modelPath);
            }

            String runnerId = id != null ? id : detectedFormat.name().toLowerCase() + "-" + modelPath.getFileName();

            // Delegate to registry for runner creation
            return ModelRunnerRegistry.get().createRunner(runnerId, detectedFormat, modelPath, device, options);
        }

        /**
         * Detects model format from file extension or directory contents.
         */
        private ModelFormat detectFormat(Path path) {
            String fileName = path.getFileName().toString().toLowerCase();

            if (fileName.endsWith(".gguf")) return ModelFormat.GGUF;
            if (fileName.endsWith(".onnx")) return ModelFormat.ONNX;
            if (fileName.endsWith(".tflite") || fileName.endsWith(".litertlm")) return ModelFormat.LITERT;
            if (fileName.endsWith(".pt") || fileName.endsWith(".pth")) return ModelFormat.TORCHSCRIPT;
            if (fileName.endsWith(".plan") || fileName.endsWith(".engine")) return ModelFormat.TENSORRT;
            if (fileName.endsWith(".safetensors")) return ModelFormat.SAFETENSORS;
            if (fileName.endsWith(".bin") || fileName.endsWith(".ckpt")) return ModelFormat.PYTORCH;

            // Check directory for marker files
            if (java.nio.file.Files.isDirectory(path)) {
                try (var stream = java.nio.file.Files.list(path)) {
                    var files = stream.toList();
                    if (files.stream().anyMatch(p -> p.toString().endsWith(".gguf"))) return ModelFormat.GGUF;
                    if (files.stream().anyMatch(p -> p.toString().endsWith(".onnx"))) return ModelFormat.ONNX;
                    if (files.stream().anyMatch(p -> p.toString().endsWith(".tflite"))) return ModelFormat.LITERT;
                    if (files.stream().anyMatch(p -> p.toString().endsWith(".safetensors"))) return ModelFormat.SAFETENSORS;
                } catch (Exception e) {
                    // Ignore
                }
            }

            return null;
        }
    }
}
