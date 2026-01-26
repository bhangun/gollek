package tech.kayys.golek.api.model;

/**
 * Device type enumeration with capability flags
 */
public enum DeviceType {
    CPU(false, false),
    CUDA(true, false),
    ROCM(true, false),
    METAL(true, false),
    TPU(false, true),
    OPENVINO(true, false);

    private final boolean supportsGpu;
    private final boolean supportsTpu;

    DeviceType(boolean supportsGpu, boolean supportsTpu) {
        this.supportsGpu = supportsGpu;
        this.supportsTpu = supportsTpu;
    }

    public boolean supportsGpu() {
        return supportsGpu;
    }

    public boolean supportsTpu() {
        return supportsTpu;
    }
}