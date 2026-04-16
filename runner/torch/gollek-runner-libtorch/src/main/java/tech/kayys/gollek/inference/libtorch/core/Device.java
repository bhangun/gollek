package tech.kayys.gollek.inference.libtorch.core;

import java.util.Objects;

/**
 * Immutable device representation (CPU, CUDA, MPS, etc.).
 * Mirrors libtorch::Device from the C++ API.
 */
public final class Device {

    /** Device type enumeration. */
    public enum Type {
        CPU(0),
        CUDA(1),
        MKLDNN(2),
        OPENGL(3),
        OPENCL(4),
        IDEEP(5),
        HIP(6),
        FPGA(7),
        MSNPU(8),
        XLA(9),
        VULKAN(10),
        METAL(11),
        XPU(12),
        MPS(13);

        private final int code;

        Type(int code) {
            this.code = code;
        }

        /** Native device type code. */
        public int code() {
            return code;
        }

        /**
         * Look up Device.Type by its native code.
         */
        public static Type fromCode(int code) {
            for (Type t : values()) {
                if (t.code == code)
                    return t;
            }
            throw new IllegalArgumentException("Unknown device type code: " + code);
        }
    }

    // Common device constants
    public static final Device CPU = new Device(Type.CPU, -1);
    public static final Device CUDA = new Device(Type.CUDA, 0);
    public static final Device MPS = new Device(Type.MPS, 0);

    private final Type type;
    private final int index;

    /**
     * Create a device with the given type and index.
     *
     * @param type  device type
     * @param index device index (-1 for default)
     */
    public Device(Type type, int index) {
        this.type = Objects.requireNonNull(type, "Device type must not be null");
        this.index = index;
    }

    /**
     * Parse a device string like "cpu", "cuda:0", "mps".
     *
     * @param deviceString device string representation
     * @return parsed Device
     * @throws IllegalArgumentException if the string cannot be parsed
     */
    public static Device parse(String deviceString) {
        Objects.requireNonNull(deviceString, "Device string must not be null");
        String[] parts = deviceString.strip().split(":");
        Type type = Type.valueOf(parts[0].toUpperCase());
        int index = parts.length > 1 ? Integer.parseInt(parts[1]) : -1;
        return new Device(type, index);
    }

    /**
     * Create a CUDA device with the given index.
     */
    public static Device cuda(int index) {
        return new Device(Type.CUDA, index);
    }

    public Type type() {
        return type;
    }

    public int index() {
        return index;
    }

    /** Check if this is a CPU device. */
    public boolean isCpu() {
        return type == Type.CPU;
    }

    /** Check if this is a CUDA device. */
    public boolean isCuda() {
        return type == Type.CUDA;
    }

    @Override
    public String toString() {
        if (index >= 0) {
            return type.name().toLowerCase() + ":" + index;
        }
        return type.name().toLowerCase();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (!(obj instanceof Device other))
            return false;
        return type == other.type && index == other.index;
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, index);
    }
}
