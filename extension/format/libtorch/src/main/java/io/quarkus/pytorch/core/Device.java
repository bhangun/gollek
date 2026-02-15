package io.github.pytorch.core;

/**
 * Device representation (CPU, CUDA, etc.)
 */
public class Device {
    
    public enum Type {
        CPU,
        CUDA,
        MKLDNN,
        OPENGL,
        OPENCL,
        IDEEP,
        HIP,
        FPGA,
        MSNPU,
        XLA,
        VULKAN,
        METAL,
        XPU,
        MPS
    }
    
    private final Type type;
    private final int index;
    
    public static final Device CPU = new Device(Type.CPU, -1);
    public static final Device CUDA = new Device(Type.CUDA, 0);
    
    public Device(Type type, int index) {
        this.type = type;
        this.index = index;
    }
    
    public Device(String deviceString) {
        String[] parts = deviceString.split(":");
        this.type = Type.valueOf(parts[0].toUpperCase());
        this.index = parts.length > 1 ? Integer.parseInt(parts[1]) : -1;
    }
    
    public Type type() {
        return type;
    }
    
    public int index() {
        return index;
    }
    
    public static Device cuda(int index) {
        return new Device(Type.CUDA, index);
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
        if (this == obj) return true;
        if (!(obj instanceof Device)) return false;
        Device other = (Device) obj;
        return type == other.type && index == other.index;
    }
    
    @Override
    public int hashCode() {
        return type.hashCode() * 31 + index;
    }
}
