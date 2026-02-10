package tech.kayys.golek.engine.model;

import java.util.Objects;

/**
 * Describes capabilities and features of a model runner.
 *
 * @author Bhangun
 * @since 1.0.0
 */
public class RunnerCapabilities {

    /**
     * Whether runner supports streaming inference.
     */
    private boolean supportsStreaming = false;

    /**
     * Whether runner supports batch inference.
     */
    private boolean supportsBatching = true;

    /**
     * Whether runner supports quantization.
     */
    private boolean supportsQuantization = false;

    /**
     * Maximum batch size supported.
     */
    private int maxBatchSize = 1;

    /**
     * Supported tensor data types.
     */
    private String[] supportedDataTypes;

    public RunnerCapabilities() {}

    public RunnerCapabilities(boolean supportsStreaming, boolean supportsBatching, boolean supportsQuantization, 
                             int maxBatchSize, String[] supportedDataTypes) {
        this.supportsStreaming = supportsStreaming;
        this.supportsBatching = supportsBatching;
        this.supportsQuantization = supportsQuantization;
        this.maxBatchSize = maxBatchSize;
        this.supportedDataTypes = supportedDataTypes;
    }

    // Getters
    public boolean isSupportsStreaming() {
        return supportsStreaming;
    }

    public boolean isSupportsBatching() {
        return supportsBatching;
    }

    public boolean isSupportsQuantization() {
        return supportsQuantization;
    }

    public int getMaxBatchSize() {
        return maxBatchSize;
    }

    public String[] getSupportedDataTypes() {
        return supportedDataTypes;
    }

    // Setters
    public void setSupportsStreaming(boolean supportsStreaming) {
        this.supportsStreaming = supportsStreaming;
    }

    public void setSupportsBatching(boolean supportsBatching) {
        this.supportsBatching = supportsBatching;
    }

    public void setSupportsQuantization(boolean supportsQuantization) {
        this.supportsQuantization = supportsQuantization;
    }

    public void setMaxBatchSize(int maxBatchSize) {
        this.maxBatchSize = maxBatchSize;
    }

    public void setSupportedDataTypes(String[] supportedDataTypes) {
        this.supportedDataTypes = supportedDataTypes;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RunnerCapabilities that = (RunnerCapabilities) o;
        return supportsStreaming == that.supportsStreaming &&
               supportsBatching == that.supportsBatching &&
               supportsQuantization == that.supportsQuantization &&
               maxBatchSize == that.maxBatchSize &&
               java.util.Arrays.equals(supportedDataTypes, that.supportedDataTypes);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(supportsStreaming, supportsBatching, supportsQuantization, maxBatchSize);
        result = 31 * result + java.util.Arrays.hashCode(supportedDataTypes);
        return result;
    }
    
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private boolean supportsStreaming = false;
        private boolean supportsBatching = true;
        private boolean supportsQuantization = false;
        private int maxBatchSize = 1;
        private String[] supportedDataTypes;

        public Builder supportsStreaming(boolean supportsStreaming) {
            this.supportsStreaming = supportsStreaming;
            return this;
        }

        public Builder supportsBatching(boolean supportsBatching) {
            this.supportsBatching = supportsBatching;
            return this;
        }

        public Builder supportsQuantization(boolean supportsQuantization) {
            this.supportsQuantization = supportsQuantization;
            return this;
        }

        public Builder maxBatchSize(int maxBatchSize) {
            this.maxBatchSize = maxBatchSize;
            return this;
        }

        public Builder supportedDataTypes(String[] supportedDataTypes) {
            this.supportedDataTypes = supportedDataTypes;
            return this;
        }

        public RunnerCapabilities build() {
            return new RunnerCapabilities(supportsStreaming, supportsBatching, supportsQuantization, 
                                       maxBatchSize, supportedDataTypes);
        }
    }
}
