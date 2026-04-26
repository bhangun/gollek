package tech.kayys.gollek.ml.tensor;

/**
 * Target device for model inference and memory allocation.
 */
public enum RunnerDevice {
    /** Auto-select best available device */
    AUTO("auto"),

    /** CPU only */
    CPU("cpu"),

    /** NVIDIA CUDA GPU */
    CUDA("cuda"),

    /** Apple Metal GPU (M1/M2/M3) */
    METAL("metal"),

    /** AMD ROCm GPU */
    ROCM("rocm"),

    /** Google Cloud TPU */
    TPU("tpu"),

    /** Neural Processing Unit (mobile) */
    NPU("npu");

    private final String name;

    RunnerDevice(String name) {
        this.name = name;
    }

    public String deviceName() {
        return name;
    }

    /**
     * Checks if this device is a GPU accelerator.
     */
    public boolean isAccelerator() {
        return this == CUDA || this == METAL || this == ROCM || this == TPU || this == NPU;
    }
}
