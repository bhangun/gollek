package tech.kayys.golek.model.core;

/**
 * Hardware capabilities
 */
public class HardwareCapabilities {
    private final boolean hasCUDA;
    private final long availableMemory;
    private final int cpuCores;

    private HardwareCapabilities(Builder builder) {
        this.hasCUDA = builder.hasCUDA;
        this.availableMemory = builder.availableMemory;
        this.cpuCores = builder.cpuCores;
    }

    public boolean hasCUDA() {
        return hasCUDA;
    }

    public long getAvailableMemory() {
        return availableMemory;
    }

    public int getCpuCores() {
        return cpuCores;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private boolean hasCUDA;
        private long availableMemory;
        private int cpuCores;

        public Builder hasCUDA(boolean hasCUDA) {
            this.hasCUDA = hasCUDA;
            return this;
        }

        public Builder availableMemory(long bytes) {
            this.availableMemory = bytes;
            return this;
        }

        public Builder cpuCores(int cores) {
            this.cpuCores = cores;
            return this;
        }

        public HardwareCapabilities build() {
            return new HardwareCapabilities(this);
        }
    }
}