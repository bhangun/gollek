package tech.kayys.gollek.converter.model;

import java.nio.file.Path;

/**
 * Result of model conversion.
 * 
 * @author bhangun
 * @version 1.0.0
 */
public class ConversionResult {

    /**
     * Conversion ID.
     */
    private final long conversionId;

    /**
     * Success flag.
     */
    private final boolean success;

    /**
     * Input model information.
     */
    private final ModelInfo inputInfo;

    /**
     * Output file path.
     */
    private final Path outputPath;

    /**
     * Output file size in bytes.
     */
    private final long outputSize;

    /**
     * Conversion duration in milliseconds.
     */
    private final long durationMs;

    /**
     * Compression ratio (input size / output size).
     */
    private final double compressionRatio;

    /**
     * Error message if failed.
     */
    private final String errorMessage;

    public ConversionResult(long conversionId, boolean success, ModelInfo inputInfo, Path outputPath, long outputSize,
            long durationMs, double compressionRatio, String errorMessage) {
        this.conversionId = conversionId;
        this.success = success;
        this.inputInfo = inputInfo;
        this.outputPath = outputPath;
        this.outputSize = outputSize;
        this.durationMs = durationMs;
        this.compressionRatio = compressionRatio;
        this.errorMessage = errorMessage;
    }

    public long getConversionId() {
        return conversionId;
    }

    public boolean isSuccess() {
        return success;
    }

    public ModelInfo getInputInfo() {
        return inputInfo;
    }

    public Path getOutputPath() {
        return outputPath;
    }

    public long getOutputSize() {
        return outputSize;
    }

    public long getDurationMs() {
        return durationMs;
    }

    public double getCompressionRatio() {
        return compressionRatio;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     * Get output size in GB.
     * 
     * @return size in GB
     */
    public double getOutputSizeGb() {
        return outputSize / (1024.0 * 1024.0 * 1024.0);
    }

    /**
     * Get formatted output size.
     * 
     * @return formatted string
     */
    public String getOutputSizeFormatted() {
        double gb = getOutputSizeGb();
        if (gb >= 1.0) {
            return String.format("%.2f GB", gb);
        }
        double mb = outputSize / (1024.0 * 1024.0);
        return String.format("%.2f MB", mb);
    }

    /**
     * Get formatted duration.
     * 
     * @return formatted string
     */
    public String getDurationFormatted() {
        long seconds = durationMs / 1000;
        if (seconds < 60) {
            return seconds + "s";
        }
        long minutes = seconds / 60;
        seconds = seconds % 60;
        return String.format("%dm %ds", minutes, seconds);
    }

    /**
     * Builder class for ConversionResult.
     */
    public static class Builder {
        private long conversionId;
        private boolean success;
        private ModelInfo inputInfo;
        private Path outputPath;
        private long outputSize;
        private long durationMs;
        private double compressionRatio;
        private String errorMessage;

        public Builder conversionId(long conversionId) {
            this.conversionId = conversionId;
            return this;
        }

        public Builder success(boolean success) {
            this.success = success;
            return this;
        }

        public Builder inputInfo(ModelInfo inputInfo) {
            this.inputInfo = inputInfo;
            return this;
        }

        public Builder outputPath(Path outputPath) {
            this.outputPath = outputPath;
            return this;
        }

        public Builder outputSize(long outputSize) {
            this.outputSize = outputSize;
            return this;
        }

        public Builder durationMs(long durationMs) {
            this.durationMs = durationMs;
            return this;
        }

        public Builder compressionRatio(double compressionRatio) {
            this.compressionRatio = compressionRatio;
            return this;
        }

        public Builder errorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
            return this;
        }

        public ConversionResult build() {
            return new ConversionResult(conversionId, success, inputInfo, outputPath, outputSize, durationMs, compressionRatio, errorMessage);
        }
    }

    public static Builder builder() {
        return new Builder();
    }
}