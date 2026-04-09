package tech.kayys.gollek.runtime.tensor;

/**
 * Target compute device for tensor operations in the Gollek inference runtime.
 * <p>
 * This enum defines the supported hardware devices where tensors can reside
 * and computations can execute. Each device type has distinct memory spaces,
 * performance characteristics, and backend support.
 * <p>
 * <h2>Device Types</h2>
 * <h3>CPU</h3>
 * <ul>
 *   <li>{@link #CPU} — General-purpose processor with system memory access</li>
 * </ul>
 * <h3>GPU Accelerators</h3>
 * <ul>
 *   <li>{@link #CUDA} — NVIDIA GPUs with CUDA support (discrete and integrated)</li>
 *   <li>{@link #METAL} — Apple GPUs via Metal API (M1/M2/M3 chips)</li>
 *   <li>{@link #ROCM} — AMD GPUs via ROCm platform</li>
 * </ul>
 * <h3>AI Accelerators</h3>
 * <ul>
 *   <li>{@link #TPU} — Google Tensor Processing Units</li>
 *   <li>{@link #NPU} — Neural Processing Units (mobile/edge AI chips)</li>
 * </ul>
 * <p>
 * <h2>Memory Spaces</h2>
 * <p>
 * Different devices have separate memory spaces:
 * </p>
 * <ul>
 *   <li><strong>CPU:</strong> System RAM, accessible by all devices via PCIe</li>
 *   <li><strong>GPU:</strong> Device VRAM, requires explicit data transfer</li>
 *   <li><strong>Unified Memory:</strong> Some platforms (Apple Silicon) support
 *       zero-copy sharing between CPU and GPU</li>
 * </ul>
 * <p>
 * <h2>Device Placement</h2>
 * <p>
 * Operations typically require all operand tensors to be on the same device.
 * Cross-device operations may require explicit data transfer:
 * </p>
 * <pre>{@code
 * // Create tensor on CPU
 * Tensor cpuTensor = backend.createTensor(shape, dtype, Device.CPU, ctx);
 * 
 * // For GPU execution, transfer data
 * Tensor gpuTensor = transfer(cpuTensor, Device.CUDA);
 * Tensor result = gpuTensor.matmul(other, ctx);
 * }</pre>
 * <p>
 * <h2>Backend Support Matrix</h2>
 * <p>
 * Not all backends support all devices:
 * </p>
 * <ul>
 *   <li><strong>LibTorch:</strong> CPU, CUDA, ROCM</li>
 *   <li><strong>GGML:</strong> CPU, METAL, CUDA</li>
 *   <li><strong>ONNX Runtime:</strong> CPU, CUDA, TPU, NPU</li>
 *   <li><strong>LiteRT:</strong> CPU, CUDA, TPU</li>
 * </ul>
 *
 * @see Tensor
 * @see Backend
 * @since 1.0
 */
public enum Device {

    /**
     * Central Processing Unit (CPU).
     * <p>
     * General-purpose processor with access to system memory. Provides
     * maximum compatibility but lower parallel throughput than accelerators.
     * </p>
     */
    CPU("cpu"),

    /**
     * NVIDIA CUDA-enabled GPU.
     * <p>
     * NVIDIA graphics processors supporting CUDA parallel computing platform.
     * Provides high-throughput parallel computation for matrix operations.
     * </p>
     */
    CUDA("cuda"),

    /**
     * Apple Metal GPU.
     * <p>
     * Apple's Metal API for GPU acceleration on macOS/iOS devices.
     * Supports Apple Silicon (M1/M2/M3) and AMD GPUs in Intel Macs.
     * </p>
     */
    METAL("metal"),

    /**
     * AMD ROCm GPU.
     * <p>
     * AMD's ROCm platform for GPU compute on Radeon graphics cards.
     * Open-source alternative to CUDA for AMD hardware.
     * </p>
     */
    ROCM("rocm"),

    /**
     * Google Tensor Processing Unit (TPU).
     * <p>
     * Google's custom ASIC designed specifically for neural network
     * machine learning. Available via Google Cloud and in Pixel devices.
     * </p>
     */
    TPU("tpu"),

    /**
     * Neural Processing Unit (NPU).
     * <p>
     * Dedicated AI accelerator found in mobile SoCs and edge devices.
     * Optimized for low-power inference workloads.
     * </p>
     */
    NPU("npu");

    /**
     * Human-readable identifier for this device.
     */
    private final String label;

    Device(String label) {
        this.label = label;
    }

    /**
     * Returns whether this device is a GPU-class hardware accelerator.
     * <p>
     * Accelerators (CUDA, METAL, ROCM, TPU, NPU) provide parallel
     * computation capabilities optimized for matrix operations but
     * require explicit data transfer from system memory.
     * </p>
     * <p>
     * {@link #CPU} returns {@code false} as it is not a specialized
     * accelerator.
     * </p>
     *
     * @return true if this device is a hardware accelerator
     */
    public boolean isAccelerator() {
        return this != CPU;
    }

    @Override
    public String toString() {
        return label;
    }
}
