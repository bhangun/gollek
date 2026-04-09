package tech.kayys.gollek.runtime.tensor;

/**
 * Identifies which native inference backend owns a tensor's memory and executes operations.
 * <p>
 * This enum serves as a unique identifier for backend implementations in the Gollek
 * inference runtime. Each backend (LibTorch, GGML, ONNX Runtime, LiteRT) registers
 * itself with the {@link BackendRegistry} under its corresponding type.
 * <p>
 * <h2>Backend Types</h2>
 * <ul>
 *   <li>{@link #LIBTORCH} — PyTorch's C++ library for deep learning inference</li>
 *   <li>{@link #GGML} — Tensor library for running LLMs on edge devices</li>
 *   <li>{@link #ONNX} — Microsoft's ONNX Runtime for cross-platform inference</li>
 *   <li>{@link #LITERT} — Google's LiteRT (formerly TFLite) for mobile/edge</li>
 *   <li>{@link #CPU_JAVA} — Pure Java implementation using Vector API</li>
 * </ul>
 * <p>
 * <h2>Backend Selection</h2>
 * <p>
 * The runtime selects backends based on:
 * </p>
 * <ul>
 *   <li>Model format compatibility (e.g., GGML for GGUF models)</li>
 *   <li>Device availability (e.g., CUDA backend for GPU execution)</li>
 *   <li>Performance characteristics (e.g., LibTorch for GPU-optimized models)</li>
 *   <li>Platform constraints (e.g., LiteRT for Android/iOS)</li>
 * </ul>
 * <p>
 * <h2>Usage</h2>
 * <pre>{@code
 * // Check backend availability
 * if (BackendRegistry.isAvailable(BackendType.GGML)) {
 *     Backend backend = BackendRegistry.get(BackendType.GGML);
 *     Tensor tensor = backend.createTensor(shape, dtype, device, ctx);
 * }
 * 
 * // Get tensor's backend
 * BackendType type = tensor.backend();
 * System.out.println("Tensor owned by: " + type.displayName());
 * }</pre>
 *
 * @see Backend
 * @see BackendRegistry
 * @see Tensor#backend()
 * @since 1.0
 */
public enum BackendType {

    /**
     * PyTorch LibTorch C++ library.
     * <p>
     * Official PyTorch inference backend providing comprehensive operator
     * support and GPU acceleration via CUDA/ROCM.
     */
    LIBTORCH("LibTorch"),

    /**
     * GGML tensor library for Large Language Models.
     * <p>
     * Lightweight C library optimized for running LLMs on consumer hardware.
     * Supports aggressive quantization (INT4, QINT4) and runs on CPU/Metal/CUDA.
     */
    GGML("GGML"),

    /**
     * Microsoft ONNX Runtime.
     * <p>
     * Cross-platform inference engine for ONNX model format.
     * Supports multiple execution providers (CPU, CUDA, TensorRT, OpenVINO).
     */
    ONNX("ONNX Runtime"),

    /**
     * Google LiteRT (formerly TensorFlow Lite).
     * <p>
     * Lightweight inference engine for mobile and edge devices.
     * Supports Android NNAPI, iOS CoreML, and embedded Linux.
     */
    LITERT("LiteRT"),

    /**
     * Pure Java CPU backend using Vector API.
     * <p>
     * Fallback implementation using Java's Panama Vector API for
     * SIMD-accelerated CPU operations without native dependencies.
     */
    CPU_JAVA("CPU (Java Vector API)");

    /**
     * Human-readable display name for this backend type.
     */
    private final String displayName;

    BackendType(String displayName) {
        this.displayName = displayName;
    }

    /**
     * Returns the human-readable display name for this backend.
     * <p>
     * The display name is suitable for logging, error messages, and
     * user-facing diagnostics.
     * </p>
     *
     * @return the display name
     */
    public String displayName() {
        return displayName;
    }

    @Override
    public String toString() {
        return displayName;
    }
}
