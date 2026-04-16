package tech.kayys.gollek.spi.tensor;

import tech.kayys.gollek.spi.model.DeviceType;

import java.lang.foreign.MemorySegment;

/**
 * Unified compute kernel SPI that normalizes hardware backends (CUDA, Metal, ROCm, DirectML).
 * <p>
 * This interface provides a hardware-agnostic contract for core LLM compute operations,
 * allowing runners and higher-level abstractions to work against a unified interface
 * rather than platform-specific binding classes.
 * <p>
 * <b>Supported Operations:</b>
 * <ul>
 *   <li>Memory management (allocation, copy, free)</li>
 *   <li>Matrix multiplication (GEMM)</li>
 *   <li>Attention computation (standard, flash, paged)</li>
 *   <li>Normalization (RMS Norm, Layer Norm)</li>
 *   <li>Activation functions (SiLU, GeLU, ReLU)</li>
 *   <li>Stream management (async execution)</li>
 * </ul>
 * 
 * <h2>Implementation Pattern</h2>
 * <pre>
 * // Platform-specific binding implements this interface
 * public class CudaBinding implements ComputeKernel {
 *     public DeviceType deviceType() { return DeviceType.CUDA; }
 *     
 *     public void matmul(...) {
 *         // Call cuBLAS via FFM
 *     }
 * }
 * 
 * public class MetalBinding implements ComputeKernel {
 *     public DeviceType deviceType() { return DeviceType.METAL; }
 *     
 *     public void matmul(...) {
 *         // Call Metal Framework via FFM
 *     }
 * }
 * </pre>
 * 
 * @see DeviceType
 * @see KernelStream
 * @since 0.1.0
 */
public interface ComputeKernel {

    // ── Device Information ──────────────────────────────────────────────

    /**
     * Gets the device type this kernel targets.
     */
    DeviceType deviceType();

    /**
     * Gets the device name (e.g., "NVIDIA A100", "Apple M2 Ultra").
     */
    String deviceName();

    /**
     * Gets the total device memory in bytes.
     */
    long totalMemory();

    /**
     * Gets the available device memory in bytes.
     */
    long availableMemory();

    /**
     * Checks if this device is available and ready.
     */
    boolean isAvailable();

    // ── Memory Management ───────────────────────────────────────────────

    /**
     * Allocates device memory.
     *
     * @param bytes number of bytes to allocate
     * @return memory segment on device
     */
    MemorySegment allocate(long bytes);

    /**
     * Allocates unified memory (CPU/GPU accessible).
     *
     * @param bytes number of bytes to allocate
     * @return unified memory segment
     */
    default MemorySegment allocateUnified(long bytes) {
        throw new UnsupportedOperationException("Unified memory not supported by " + deviceType());
    }

    /**
     * Frees device memory.
     *
     * @param ptr memory segment to free
     */
    void free(MemorySegment ptr);

    /**
     * Copies data from host to device.
     *
     * @param dst destination (device)
     * @param src source (host)
     * @param bytes number of bytes to copy
     */
    void copyHostToDevice(MemorySegment dst, MemorySegment src, long bytes);

    /**
     * Copies data from device to host.
     *
     * @param dst destination (host)
     * @param src source (device)
     * @param bytes number of bytes to copy
     */
    void copyDeviceToHost(MemorySegment dst, MemorySegment src, long bytes);

    /**
     * Copies data within device memory.
     *
     * @param dst destination (device)
     * @param src source (device)
     * @param bytes number of bytes to copy
     */
    void copyDeviceToDevice(MemorySegment dst, MemorySegment src, long bytes);

    // ── Matrix Multiplication ───────────────────────────────────────────

    /**
     * Performs matrix multiplication: C = A × B
     * <p>
     * Matrix dimensions: A[M×K] × B[K×N] = C[M×N]
     *
     * @param C output matrix [M×N]
     * @param A input matrix [M×K]
     * @param B input matrix [K×N]
     * @param M rows in A and C
     * @param K cols in A, rows in B
     * @param N cols in B and C
     */
    void matmul(MemorySegment C, MemorySegment A, MemorySegment B, int M, int K, int N);

    /**
     * Performs matrix multiplication with bias add: C = A × B + C
     * <p>
     * Matrix dimensions: A[M×K] × B[K×N] + C[M×N] = C[M×N]
     *
     * @param C output/bias matrix [M×N]
     * @param A input matrix [M×K]
     * @param B input matrix [K×N]
     * @param M rows in A and C
     * @param K cols in A, rows in B
     * @param N cols in B and C
     */
    default void matmulAdd(MemorySegment C, MemorySegment A, MemorySegment B, int M, int K, int N) {
        // Default: compute A×B to temp, then add to C
        MemorySegment temp = allocate((long) M * N * Float.BYTES);
        try {
            matmul(temp, A, B, M, K, N);
            elementwiseAdd(C, C, temp, (long) M * N);
        } finally {
            free(temp);
        }
    }

    // ── Attention Computation ───────────────────────────────────────────

    /**
     * Performs standard scaled dot-product attention.
     * <p>
     * Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
     *
     * @param output output tensor [seq_len, d_model]
     * @param query query tensor [seq_len, num_heads, head_dim]
     * @param key key tensor [seq_len, num_heads, head_dim]
     * @param value value tensor [seq_len, num_heads, head_dim]
     * @param seqLen sequence length
     * @param numHeads number of attention heads
     * @param headDim head dimension
     */
    void attention(MemorySegment output, MemorySegment query, MemorySegment key, 
                   MemorySegment value, int seqLen, int numHeads, int headDim);

    /**
     * Performs flash attention (memory-efficient attention).
     * <p>
     * Uses tiling to reduce memory complexity from O(N²) to O(N).
     *
     * @param output output tensor
     * @param query query tensor
     * @param key key tensor
     * @param value value tensor
     * @param seqLen sequence length
     * @param numHeads number of heads
     * @param headDim head dimension
     */
    default void flashAttention(MemorySegment output, MemorySegment query, MemorySegment key,
                               MemorySegment value, int seqLen, int numHeads, int headDim) {
        // Default fallback to standard attention
        attention(output, query, key, value, seqLen, numHeads, headDim);
    }

    // ── Normalization ───────────────────────────────────────────────────

    /**
     * Performs RMS normalization.
     * <p>
     * RMSNorm(x) = x / sqrt(mean(x²) + ε) × weight
     *
     * @param output output tensor
     * @param input input tensor
     * @param weight weight tensor
     * @param hiddenSize hidden dimension size
     * @param eps epsilon for numerical stability
     */
    void rmsNorm(MemorySegment output, MemorySegment input, MemorySegment weight, 
                 int hiddenSize, float eps);

    // ── Activation Functions ────────────────────────────────────────────

    /**
     * Applies SiLU activation: x × sigmoid(x)
     *
     * @param output output tensor
     * @param input input tensor
     * @param size number of elements
     */
    void silu(MemorySegment output, MemorySegment input, long size);

    /**
     * Applies SiLU FFN: x × sigmoid(x) (optimized for feed-forward networks)
     *
     * @param output output tensor
     * @param input input tensor
     * @param size number of elements
     */
    default void siluFfn(MemorySegment output, MemorySegment input, long size) {
        silu(output, input, size);
    }

    /**
     * Applies GeLU activation.
     *
     * @param output output tensor
     * @param input input tensor
     * @param size number of elements
     */
    default void gelu(MemorySegment output, MemorySegment input, long size) {
        throw new UnsupportedOperationException("GeLU not supported by " + deviceType());
    }

    /**
     * Applies ReLU activation: max(0, x)
     *
     * @param output output tensor
     * @param input input tensor
     * @param size number of elements
     */
    default void relu(MemorySegment output, MemorySegment input, long size) {
        throw new UnsupportedOperationException("ReLU not supported by " + deviceType());
    }

    // ── Element-wise Operations ─────────────────────────────────────────

    /**
     * Performs element-wise addition: C = A + B
     *
     * @param C output tensor
     * @param A input tensor A
     * @param B input tensor B
     * @param size number of elements
     */
    void elementwiseAdd(MemorySegment C, MemorySegment A, MemorySegment B, long size);

    /**
     * Performs element-wise multiplication: C = A × B
     *
     * @param C output tensor
     * @param A input tensor A
     * @param B input tensor B
     * @param size number of elements
     */
    void elementwiseMul(MemorySegment C, MemorySegment A, MemorySegment B, long size);

    /**
     * Performs element-wise scaling: A = A × scale
     *
     * @param A input/output tensor
     * @param scale scale factor
     * @param size number of elements
     */
    void elementwiseScale(MemorySegment A, float scale, long size);

    // ── Stream Management ───────────────────────────────────────────────

    /**
     * Creates a new execution stream for async operations.
     *
     * @return stream handle
     */
    default KernelStream createStream() {
        throw new UnsupportedOperationException("Streams not supported by " + deviceType());
    }

    /**
     * Synchronizes on a stream (waits for completion).
     *
     * @param stream stream to synchronize
     */
    default void synchronizeStream(KernelStream stream) {
        throw new UnsupportedOperationException("Streams not supported by " + deviceType());
    }

    // ── Lifecycle ───────────────────────────────────────────────────────

    /**
     * Initializes the kernel and device.
     */
    default void initialize() {
        // Default: no initialization needed
    }

    /**
     * Shuts down the kernel and releases device resources.
     */
    default void shutdown() {
        // Default: no cleanup needed
    }

    // ── Nested Types ────────────────────────────────────────────────────

    /**
     * Opaque handle to an execution stream.
     */
    interface KernelStream {
        void close();
    }
}
