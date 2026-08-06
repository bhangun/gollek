/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.plugin.runner;

import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletionStage;
import io.smallrye.mutiny.Uni;

/**
 * Enhanced SPI for model runner plugins with comprehensive lifecycle
 * management,
 * error handling, and observability support.
 *
 * <p>
 * Runner plugins provide support for different model formats:
 * <ul>
 * <li><b>GGUF</b> - llama.cpp format with quantization support</li>
 * <li><b>ONNX</b> - Open Neural Network Exchange format</li>
 * <li><b>Safetensors</b> - Hugging Face safe tensor format</li>
 * <li><b>TensorRT</b> - NVIDIA TensorRT optimized engines</li>
 * <li><b>LibTorch</b> - PyTorch C++ API format</li>
 * <li><b>TFLite</b> - TensorFlow Lite format</li>
 * </ul>
 *
 * <h2>Plugin Lifecycle</h2>
 * 
 * <pre>
 * ┌─────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐
 * │ LOADED  │ →  │ VALIDATING│ →  │ VALIDATED │ →  │INITIALIZING│ →  │ ACTIVE  │
 * └─────────┘    └──────────┘    └───────────┘    └──────────┘    └─────────┘
 *                                                               ↓
 *                                                          ┌─────────┐
 *                                                          │ ERROR   │
 *                                                          └─────────┘
 *                                                               ↓
 *                                                          ┌─────────┐
 *                                                          │STOPPED  │
 *                                                          └─────────┘
 * </pre>
 *
 * <h2>Example Implementation</h2>
 * 
 * <pre>{@code
 * public class GGUFRunnerPlugin implements RunnerPlugin {
 *
 *     {@code @Override}
 *     public String id() { return "gguf-runner"; }
 *
 *     {@code @Override}
 *     public RunnerConfig getConfig() {
 *         return RunnerConfig.builder()
 *             .nGpuLayers(-1)
 *             .contextSize(4096)
 *             .threads(4)
 *             .build();
 *     }
 *
 *     {@code @Override}
 *     public void initialize(RunnerContext context) throws RunnerException {
 *         // Initialize llama.cpp backend
 *         initLlamaCpp(context.getConfig());
 *     }
 *
 *     {@code @Override}
 *     public <T> RunnerResult<T> execute(RunnerRequest request,
 *                                        RunnerContext context) {
 *         return switch (request.getType()) {
 *             case INFER -> executeInference(request, context);
 *             case EMBED -> executeEmbedding(request, context);
 *             default -> throw new UnknownRequestException(request.getType());
 *         };
 *     }
 * }
 * }</pre>
 *
 * <h2>Format Detection</h2>
 * 
 * <pre>{@code
 * // Auto-detect model format
 * String format = RunnerPlugin.detectFormat(modelPath);
 *
 * // Load appropriate runner plugin
 * RunnerPlugin runner = RunnerPluginManager.getInstance()
 *         .getPluginForFormat(format)
 *         .orElseThrow(() -> new RunnerNotFoundException(format));
 * }</pre>
 *
 * @since 2.1.0
 * @version 2.0.0 (Enhanced with lifecycle, error handling, and observability)
 */
public interface RunnerPlugin {

    // ========================================================================
    // Plugin Identity
    // ========================================================================

    /**
     * Unique runner plugin identifier.
     *
     * @return plugin ID (e.g., "gguf-runner", "onnx-runner", "safetensor-runner")
     */
    String id();

    /**
     * Human-readable name.
     *
     * @return runner name (e.g., "GGUF Runner", "ONNX Runtime Runner")
     */
    String name();

    /**
     * Runner version following semantic versioning.
     *
     * @return version string (e.g., "1.0.0", "2.1.3")
     */
    default String version() {
        return "1.0.0";
    }

    /**
     * Runner description.
     *
     * @return detailed description of supported formats and capabilities
     */
    String description();

    /**
     * Get the primary format for this runner.
     *
     * @return format ID (e.g., "gguf", "onnx", "safetensors")
     */
    String format();

    // ========================================================================
    // Lifecycle Management
    // ========================================================================

    /**
     * Validate runner prerequisites before initialization.
     *
     * <p>
     * Called during plugin validation phase to ensure all requirements
     * are met (libraries, backends, hardware).
     * </p>
     *
     * @return validation result with status and any error messages
     */
    default RunnerValidationResult validate() {
        try {
            boolean available = isAvailable();
            if (available) {
                return RunnerValidationResult.valid();
            } else {
                return RunnerValidationResult.invalid(
                        "Runner not available on this platform");
            }
        } catch (Exception e) {
            return RunnerValidationResult.invalid(
                    "Validation failed: " + e.getMessage());
        }
    }

    /**
     * Initialize the runner with context and configuration.
     *
     * <p>
     * Called during plugin initialization phase. Should prepare all
     * resources needed for model loading and inference.
     * </p>
     *
     * @param context Runner context with configuration and services
     * @throws RunnerException if initialization fails
     */
    default void initialize(RunnerContext context) throws RunnerException {
        // Default: no-op
    }

    /**
     * Check if this runner is available on current platform.
     *
     * <p>
     * Performs quick availability check without detailed validation.
     * </p>
     *
     * @return true if runner can potentially be used
     */
    boolean isAvailable();

    /**
     * Check if runner is healthy and operational.
     *
     * <p>
     * Called periodically during runtime to monitor runner health.
     * </p>
     *
     * @return true if runner is healthy and ready for operations
     */
    default boolean isHealthy() {
        return true;
    }

    /**
     * Get detailed health information.
     *
     * @return health status with details
     */
    default RunnerHealth health() {
        boolean healthy = isHealthy();
        return healthy ? RunnerHealth.healthy() : RunnerHealth.unhealthy("Runner reported unhealthy");
    }

    /**
     * Shutdown and cleanup resources.
     *
     * <p>
     * Called when plugin is being unloaded. Should release all resources
     * and perform cleanup.
     * </p>
     */
    default void shutdown() {
        // Default: no-op
    }

    // ========================================================================
    // Model Support
    // ========================================================================

    /**
     * Supported file extensions.
     *
     * @return set of file extensions (e.g., ".gguf", ".onnx", ".safetensors")
     */
    Set<String> supportedFormats();

    /**
     * Supported model architectures.
     *
     * @return set of architecture names (e.g., "llama", "mistral", "bert", "gpt2")
     */
    Set<String> supportedArchitectures();

    /**
     * Check if this runner can handle the given model.
     *
     * @param modelPath Path to model file
     * @return true if runner supports this model
     */
    default boolean supportsModel(String modelPath) {
        String extension = getExtension(modelPath);
        return supportedFormats().contains(extension.toLowerCase());
    }

    /**
     * Get runner priority. Higher values execute first.
     *
     * @return priority value (default: 0)
     */
    default int priority() {
        return 0;
    }

    // ========================================================================
    // Inference Operations
    // ========================================================================

    /**
     * Execute a runner operation synchronously.
     *
     * @param request Operation request
     * @param context Execution context with configuration
     * @param <T>     Expected result type
     * @return execution result
     * @throws RunnerException if execution fails
     */
    <T> RunnerResult<T> execute(RunnerRequest request,
            RunnerContext context) throws RunnerException;

    /**
     * Execute a runner operation asynchronously.
     *
     * @param request Operation request
     * @param context Execution context with configuration
     * @param <T>     Expected result type
     * @return completion stage with result
     */
    default <T> CompletionStage<RunnerResult<T>> executeAsync(
            RunnerRequest request,
            RunnerContext context) {
        // Default: wrap synchronous execution
        try {
            RunnerResult<T> result = execute(request, context);
            return java.util.concurrent.CompletableFuture.completedFuture(result);
        } catch (RunnerException e) {
            java.util.concurrent.CompletableFuture<RunnerResult<T>> future = new java.util.concurrent.CompletableFuture<>();
            future.completeExceptionally(e);
            return future;
        }
    }

    // ========================================================================
    // Session Management
    // ========================================================================

    /**
     * Create a runner session for the given model (legacy method).
     *
     * @param modelPath Path to model file
     * @param config    Session configuration
     * @return Runner session
     * @deprecated Use loadModel(ModelLoadRequest, RunnerContext) instead
     */
    @Deprecated(since = "2.0.0", forRemoval = false)
    default RunnerSession createSession(String modelPath, Map<String, Object> config) {
        throw new UnsupportedOperationException(
                "Legacy createSession method not implemented. " +
                        "Use loadModel(ModelLoadRequest, RunnerContext) instead.");
    }

    /**
     * Load a model into the runner.
     *
     * @param request Model load request
     * @param context Runner context
     * @return model handle
     * @throws RunnerException if loading fails
     */
    default ModelHandle loadModel(ModelLoadRequest request, RunnerContext context)
            throws RunnerException {
        throw new UnsupportedOperationException(
                "loadModel not implemented. Use legacy createSession instead.");
    }

    /**
     * Unload a model from the runner.
     *
     * @param modelHandle Model handle to unload
     * @param context     Runner context
     * @throws RunnerException if unloading fails
     */
    default void unloadModel(ModelHandle modelHandle, RunnerContext context)
            throws RunnerException {
        // Default: no-op
    }

    // ========================================================================
    // Capabilities and Metadata
    // ========================================================================

    /**
     * Get runner-specific metadata.
     *
     * @return metadata map with capabilities, versions, and features
     */
    default Map<String, Object> metadata() {
        return Map.of(
                "format", format(),
                "version", version(),
                "formats", supportedFormats(),
                "architectures", supportedArchitectures(),
                "priority", priority());
    }

    /**
     * Get runner configuration.
     *
     * @return runner configuration
     */
    default RunnerConfig getConfig() {
        return RunnerConfig.defaultConfig();
    }

    /**
     * Get plugin dependencies.
     *
     * @return map of plugin ID to version requirement
     */
    default Map<String, String> dependencies() {
        return Collections.emptyMap();
    }

    /**
     * Get supported request types.
     *
     * @return set of supported request types
     */
    default Set<RequestType> supportedRequestTypes() {
        return Set.of(RequestType.INFER);
    }

    // ========================================================================
    // Format Detection
    // ========================================================================

    /**
     * Detect model format from file path.
     *
     * @param modelPath path to model file
     * @return detected format or "unknown"
     */
    static String detectFormat(String modelPath) {
        String extension = getExtension(modelPath);

        return switch (extension.toLowerCase()) {
            case ".gguf" -> "gguf";
            case ".onnx" -> "onnx";
            case ".safetensors", ".safetensor" -> "safetensors";
            case ".engine", ".plan" -> "tensorrt";
            case ".pt", ".pth", ".bin" -> "libtorch";
            case ".litertlm", ".tfl" -> "litert";
            default -> "unknown";
        };
    }

    /**
     * Get file extension from path.
     *
     * @param path file path
     * @return file extension (including dot)
     */
    static String getExtension(String path) {
        int lastDot = path.lastIndexOf('.');
        if (lastDot >= 0 && lastDot < path.length() - 1) {
            return path.substring(lastDot);
        }
        return "";
    }

    /**
     * Check if a class is available in the classpath.
     *
     * @param className fully qualified class name
     * @return true if class can be loaded
     */
    static boolean isClassAvailable(String className) {
        try {
            Class.forName(className);
            return true;
        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            return false;
        }
    }
}
