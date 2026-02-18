package tech.kayys.gollek.engine.model;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.HealthStatus;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.ResourceMetrics;
import tech.kayys.gollek.spi.model.RunnerMetadata;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import io.smallrye.mutiny.Multi;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.engine.exception.RunnerInitializationException;

/**
 * Core SPI for model execution backends.
 * All model runner implementations must implement this interface.
 * Service Provider Interface (SPI) for model execution backends.
 * 
 * <p>
 * All inference adapters (LiteRT, ONNX, TensorFlow, JAX, PyTorch, Triton) must
 * implement
 * this interface to integrate with the platform's routing and execution
 * framework.
 * 
 * <p>
 * Lifecycle:
 * 
 * <pre>
 * 1. initialize() - Load model artifacts and allocate resources
 * 2. infer() or inferAsync() - Execute inference requests (can be called multiple times)
 * 3. health() - Check runner availability (called periodically)
 * 4. close() - Release all resources
 * </pre>
 * 
 * <p>
 * Thread Safety: Implementations must be thread-safe for concurrent inference
 * calls.
 * 
 * <p>
 * Resource Management: Implementations should use try-with-resources pattern or
 * ensure proper cleanup in close() method.
 * 
 * <p>
 * This is the canonical interface for model runners - all other
 * versions are deprecated and will be removed in future versions.
 * </p>
 */
public interface ModelRunner extends AutoCloseable {

    /**
     * Get the unique identifier for this runner type.
     * 
     * <p>
     * Examples: "litert-cpu", "litert-gpu", "onnx-cuda", "tensorflow-tpu",
     * "triton-grpc"
     * 
     * @return Runner identifier used for logging, metrics, and selection
     */
    String name();

    /**
     * Get the framework this runner wraps.
     * 
     * <p>
     * Examples: "litert", "onnx", "tensorflow", "pytorch", "jax"
     * 
     * @return Framework name
     */
    String framework();

    /**
     * Get the execution device type.
     * 
     * @return Device type (CPU, CUDA, ROCM, TPU, NPU, etc.)
     */
    DeviceType deviceType();

    /**
     * Get runner capabilities and supported features.
     * 
     * @return Capabilities descriptor
     */
    RunnerCapabilities capabilities();

    /**
     * Get runner metrics for the last N inferences.
     * 
     * @return Runtime metrics
     */
    RunnerMetrics metrics();

    /**
     * Initialize the runner with model artifacts and configuration.
     * 
     * <p>
     * This method is called once during runner creation. It should:
     * <ul>
     * <li>Load model files from the manifest (local filesystem, S3, GCS, etc.)</li>
     * <li>Allocate hardware resources (GPU memory, TPU cores, etc.)</li>
     * <li>Initialize native libraries (via FFM for optimal performance)</li>
     * <li>Validate model compatibility with the runner</li>
     * <li>Warm up the model if configured</li>
     * </ul>
     * 
     * @param manifest Model metadata including artifact URIs, formats, and resource
     *                 requirements
     * @param config   Runner-specific configuration (device selection, batch size,
     *                 optimization flags)
     * @throws RunnerInitializationException if initialization fails
     */
    void initialize(ModelManifest manifest, RunnerConfiguration config) throws RunnerInitializationException;

    /**
     * Execute synchronous inference on the model.
     * 
     * <p>
     * This is the primary method for most inference workloads. The implementation
     * should:
     * <ul>
     * <li>Convert request tensors to the backend's native format</li>
     * <li>Execute inference with appropriate timeout handling</li>
     * <li>Convert output tensors back to platform format</li>
     * <li>Handle errors gracefully with meaningful exceptions</li>
     * </ul>
     * 
     * <p>
     * Performance Note: Implementations should avoid blocking operations where
     * possible.
     * For long-running inferences, consider using
     * {@link #inferAsync(InferenceRequest)}.
     * 
     * @param request Inference request with inputs
     * @param context Request context with timeout, priority, etc.
     * @return Inference response with outputs
     * @throws tech.kayys.gollek.spi.exception.InferenceException if execution fails
     */
    InferenceResponse infer(
            InferenceRequest request) throws InferenceException;

    /**
     * Execute asynchronous inference with callback
     * 
     * @param request Inference request
     * @return CompletionStage for async processing
     */
    /**
     * Execute asynchronous inference on the model.
     * 
     * <p>
     * This method is optimal for:
     * <ul>
     * <li>Batch processing scenarios</li>
     * <li>Long-running model executions (LLMs, diffusion models)</li>
     * <li>Pipeline architectures with multiple models</li>
     * <li>When client needs to perform other work while waiting</li>
     * </ul>
     * 
     * <p>
     * Default implementation delegates to synchronous
     * {@link #infer(InferenceRequest)}
     * but adapters should override with true async implementation when supported by
     * the backend.
     * 
     * @param request The inference request containing input tensors and metadata
     * @return CompletableFuture that completes with inference response
     */
    default CompletableFuture<InferenceResponse> inferAsync(InferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return infer(request);
            } catch (InferenceException e) {
                throw new RuntimeException(e);
            }
        });
    }

    /**
     * Check if the runner is healthy and ready to serve requests.
     * 
     * <p>
     * Health checks should verify:
     * <ul>
     * <li>Runner is initialized</li>
     * <li>Model is loaded and accessible</li>
     * <li>Hardware resources are available (GPU memory, TPU connectivity)</li>
     * <li>Native libraries are responsive</li>
     * </ul>
     * 
     * <p>
     * This method is called by:
     * <ul>
     * <li>Kubernetes liveness/readiness probes</li>
     * <li>Load balancer health checks</li>
     * <li>Router selection algorithms</li>
     * <li>Monitoring systems</li>
     * </ul>
     * 
     * @return true if runner is healthy and can process requests, false otherwise
     */
    boolean health();

    /**
     * Get detailed health information including metrics.
     * 
     * <p>
     * Default implementation returns basic health status. Adapters can override
     * to provide rich diagnostics:
     * <ul>
     * <li>Memory usage (heap, native, GPU VRAM)</li>
     * <li>Request queue depth</li>
     * <li>Average/P95/P99 latency</li>
     * <li>Error rates</li>
     * <li>Hardware utilization</li>
     * </ul>
     * 
     * @return HealthStatus with detailed metrics
     */
    default HealthStatus healthStatus() {
        return health()
                ? HealthStatus.healthy("Runner " + name() + " is operational")
                : HealthStatus.unhealthy("Runner " + name() + " is not operational");
    }

    /**
     * Get current resource utilization metrics
     * 
     * @return Resource usage snapshot
     */
    ResourceMetrics getMetrics();

    /**
     * Warm up the model (optional optimization)
     * 
     * @param sampleInputs Sample inputs for warming
     */
    default void warmup(List<InferenceRequest> sampleInputs) {
        // Default no-op
    }

    /**
     * Get runner metadata
     * 
     * @return Metadata about this runner implementation
     */
    RunnerMetadata metadata();

    /**
     * Release all resources held by this runner.
     * 
     * <p>
     * This method should:
     * <ul>
     * <li>Unload model from memory</li>
     * <li>Release GPU/TPU/NPU resources</li>
     * <li>Close native library handles</li>
     * <li>Shutdown thread pools</li>
     * <li>Clear caches</li>
     * </ul>
     * 
     * Gracefully release resources
     * <p>
     * Must be idempotent - multiple calls should be safe.
     * Must not throw exceptions - log errors instead.
     */
    /**
     * Release all resources held by this runner.
     */
    @Override
    void close();

    /**
     * Execute streaming inference
     * 
     * @param request The inference request
     * @return Multi stream of chunks
     */
    default Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().failure(new UnsupportedOperationException("Streaming not support for this runner"));
    }
}
