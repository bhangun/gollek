package tech.kayys.golek.converter.model;

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

}