package tech.kayys.golek.api.model;

/**
 * Enumeration of supported hardware acceleration devices.
 * 
 * <p>
 * The platform supports heterogeneous deployment across multiple
 * device types, with automatic selection based on availability,
 * cost, and performance requirements.
 * 
 * @author bhangun
 * @since 1.0.0
 */
public enum DeviceType {
    /**
     * CPU execution (x86_64, ARM64).
     * Always available, lowest cost, highest compatibility.
     */
    CPU("cpu", "CPU", 1, false),

    /**
     * NVIDIA CUDA GPU acceleration.
     * High throughput for parallel workloads.
     */
    CUDA("cuda", "NVIDIA CUDA GPU", 10, true),

    /**
     * AMD ROCm GPU acceleration.
     * Alternative to CUDA for AMD hardware.
     */
    ROCM("rocm", "AMD ROCm GPU", 10, true),

    /**
     * Intel GPU acceleration (Arc, Xe, integrated).
     * Emerging option for Intel-based systems.
     */
    INTEL_GPU("intel-gpu", "Intel GPU", 8, true),

    /**
     * Apple Metal GPU acceleration.
     * For Apple Silicon (M1/M2/M3) systems.
     */
    METAL("metal", "Apple Metal GPU", 9, true),

    /**
     * Google Cloud TPU (Tensor Processing Unit).
     * Optimized for TensorFlow and JAX workloads.
     */
    TPU("tpu", "Google Cloud TPU", 15, true),

    /**
     * Neural Processing Unit (NPU/VPU).
     * Edge AI accelerators: Qualcomm, MediaTek, ARM Ethos, Intel Movidius.
     */
    NPU("npu", "Neural Processing Unit", 12, true),

    /**
     * AWS Inferentia / Graviton chips.
     */
    INFERENTIA("inferentia", "AWS Inferentia", 14, true),

    /**
     * Azure Maia custom silicon.
     */
    AZURE_MAIA("azure-maia", "Azure Maia", 14, true),

    /**
     * DirectML (Windows ML acceleration layer).
     * Works across NVIDIA, AMD, and Intel GPUs on Windows.
     */
    DIRECTML("directml", "DirectML", 8, true),

    /**
     * WebGPU for browser-based inference.
     */
    WEBGPU("webgpu", "WebGPU", 5, true),

    /**
     * Remote execution via gRPC (Triton, TFServing, TorchServe).
     * Device type determined by remote server.
     */
    REMOTE("remote", "Remote gRPC", 0, false),

    /**
     * Google Cloud TPU v4.
     */
    TPU_V4("tpu-v4", "Google Cloud TPU v4", 16, true),

    /**
     * Google Cloud TPU v5.
     */
    TPU_V5("tpu-v5", "Google Cloud TPU v5", 17, true),

    /**
     * OpenVINO Intel optimization toolkit.
     */
    OPENVINO("openvino", "Intel OpenVINO", 7, true);

    private final String id;
    private final String displayName;
    private final int performanceScore; // Relative performance (higher = faster, 0 = variable)
    private final boolean requiresSpecializedHardware;

    DeviceType(String id, String displayName, int performanceScore, boolean requiresSpecializedHardware) {
        this.id = id;
        this.displayName = displayName;
        this.performanceScore = performanceScore;
        this.requiresSpecializedHardware = requiresSpecializedHardware;
    }

    /**
     * Parse device type from string identifier (case-insensitive).
     * 
     * @param id Device type identifier
     * @return Matching DeviceType
     * @throws IllegalArgumentException if no match found
     */
    public static DeviceType fromId(String id) {
        for (DeviceType type : values()) {
            if (type.id.equalsIgnoreCase(id)) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown device type: " + id);
    }

    /**
     * Check if this device type is GPU-based.
     */
    public boolean isGpu() {
        return this == CUDA || this == ROCM || this == INTEL_GPU || this == METAL || this == DIRECTML || this == WEBGPU;
    }

    /**
     * Check if this device type requires cloud infrastructure.
     */
    public boolean isCloud() {
        return this == TPU || this == TPU_V4 || this == TPU_V5 || this == INFERENTIA || this == AZURE_MAIA;
    }

    /**
     * Check if this device type is suitable for edge deployment.
     */
    public boolean isEdge() {
        return this == CPU || this == NPU || this == METAL;
    }

    /**
     * Check if this device type supports GPU acceleration.
     */
    public boolean supportsGpu() {
        return isGpu();
    }

    /**
     * Check if this device type supports TPU acceleration.
     */
    public boolean supportsTpu() {
        return this == TPU || this == TPU_V4 || this == TPU_V5;
    }

    public String getId() {
        return id;
    }

    public String getDisplayName() {
        return displayName;
    }

    public int getPerformanceScore() {
        return performanceScore;
    }

    public boolean isRequiresSpecializedHardware() {
        return requiresSpecializedHardware;
    }
}