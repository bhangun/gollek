package tech.kayys.gollek.sdk.core;

/**
 * Device abstraction for tensor placement (CPU, CUDA, MPS, etc.).
 *
 * <p>Similar to PyTorch's `torch.device`, this class represents the compute
 * device where a tensor's data resides.</p>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * Device cpu = Device.CPU;
 * Device cuda = Device.of("cuda", 0);  // GPU 0
 * Device mps = Device.of("mps");       // Apple Silicon
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public final class Device {

    /** CPU device */
    public static final Device CPU = new Device("cpu", -1);

    /** Default CUDA device (GPU 0) */
    public static final Device CUDA = new Device("cuda", 0);

    /** Apple Metal Performance Shaders (Apple Silicon) */
    public static final Device MPS = new Device("mps", -1);

    private final String type;
    private final int index;

    private Device(String type, int index) {
        this.type = type;
        this.index = index;
    }

    /**
     * Create a device.
     *
     * @param type  device type ("cpu", "cuda", "mps")
     * @param index device index (for multi-GPU, -1 for default)
     * @return device instance
     */
    public static Device of(String type, int index) {
        return new Device(type, index);
    }

    /**
     * Create a device (default index).
     *
     * @param type device type
     * @return device instance
     */
    public static Device of(String type) {
        return of(type, -1);
    }

    /**
     * Get device type.
     */
    public String type() {
        return type;
    }

    /**
     * Get device index.
     */
    public int index() {
        return index;
    }

    /**
     * Check if this is a CUDA device.
     */
    public boolean isCuda() {
        return "cuda".equals(type);
    }

    /**
     * Check if this is a CPU device.
     */
    public boolean isCpu() {
        return "cpu".equals(type);
    }

    @Override
    public String toString() {
        return index >= 0 ? String.format("%s:%d", type, index) : type;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Device device)) return false;
        return type.equals(device.type) && index == device.index;
    }

    @Override
    public int hashCode() {
        return 31 * type.hashCode() + index;
    }
}
