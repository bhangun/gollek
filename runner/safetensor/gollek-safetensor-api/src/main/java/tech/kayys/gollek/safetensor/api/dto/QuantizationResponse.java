/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantizationResponse.java
 * ───────────────────────
 * Quantization response DTO.
 */
package tech.kayys.gollek.safetensor.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Duration;

/**
 * Response payload for quantization operations.
 * DTO moved to use primitive types to avoid circular dependencies with Quantization module.
 */
public class QuantizationResponse {

    @JsonProperty("success")
    private boolean success;

    @JsonProperty("output_path")
    private String outputPath;

    @JsonProperty("original_size")
    private String originalSize;

    @JsonProperty("quantized_size")
    private String quantizedSize;

    @JsonProperty("compression_ratio")
    private double compressionRatio;

    @JsonProperty("duration_ms")
    private long durationMs;

    @JsonProperty("tensor_count")
    private int tensorCount;

    @JsonProperty("param_count")
    private long paramCount;

    @JsonProperty("avg_quant_error_mse")
    private double avgQuantErrorMse;

    @JsonProperty("error_message")
    private String errorMessage;

    @JsonProperty("strategy")
    private String strategy;

    public QuantizationResponse() {
    }

    /**
     * reflection-based conversion from QuantResult to avoid direct dependency.
     */
    public static QuantizationResponse fromReflection(Object result) {
        QuantizationResponse response = new QuantizationResponse();
        if (result == null) return error("Result object is null");

        try {
            boolean success = (boolean) result.getClass().getMethod("isSuccess").invoke(result);
            response.setSuccess(success);

            Object config = result.getClass().getMethod("getConfig").invoke(result);
            Object strategy = config.getClass().getMethod("getStrategy").invoke(config);
            response.setStrategy(strategy.toString());

            if (success) {
                Object stats = result.getClass().getMethod("getStats").invoke(result);
                Object path = result.getClass().getMethod("getQuantizedModelPath").invoke(result);
                
                response.setOutputPath(path != null ? path.toString() : null);
                
                // Helper to call QuantStats.formatSize if we can, but it's easier to just get the bytes
                long originalBytes = (long) stats.getClass().getMethod("getOriginalSizeBytes").invoke(stats);
                long quantizedBytes = (long) stats.getClass().getMethod("getQuantizedSizeBytes").invoke(stats);
                
                response.setOriginalSize(formatBytes(originalBytes));
                response.setQuantizedSize(formatBytes(quantizedBytes));
                response.setCompressionRatio((double) stats.getClass().getMethod("getCompressionRatio").invoke(stats));
                response.setDurationMs(((Duration) stats.getClass().getMethod("getDuration").invoke(stats)).toMillis());
                response.setTensorCount((int) stats.getClass().getMethod("getTensorCount").invoke(stats));
                response.setParamCount((long) stats.getClass().getMethod("getParamCount").invoke(stats));
                response.setAvgQuantErrorMse((double) stats.getClass().getMethod("getAvgQuantErrorMse").invoke(stats));
            } else {
                response.setErrorMessage((String) result.getClass().getMethod("getErrorMessage").invoke(result));
            }
        } catch (Exception e) {
            response.setSuccess(false);
            response.setErrorMessage("Error converting result: " + e.getMessage());
        }

        return response;
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        char pre = "KMGTPE".charAt(exp - 1);
        return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
    }

    public static QuantizationResponse error(String errorMessage) {
        QuantizationResponse response = new QuantizationResponse();
        response.setSuccess(false);
        response.setErrorMessage(errorMessage);
        return response;
    }

    public boolean isSuccess() { return success; }
    public void setSuccess(boolean success) { this.success = success; }
    public String getOutputPath() { return outputPath; }
    public void setOutputPath(String outputPath) { this.outputPath = outputPath; }
    public String getOriginalSize() { return originalSize; }
    public void setOriginalSize(String originalSize) { this.originalSize = originalSize; }
    public String getQuantizedSize() { return quantizedSize; }
    public void setQuantizedSize(String quantizedSize) { this.quantizedSize = quantizedSize; }
    public double getCompressionRatio() { return compressionRatio; }
    public void setCompressionRatio(double compressionRatio) { this.compressionRatio = compressionRatio; }
    public long getDurationMs() { return durationMs; }
    public void setDurationMs(long durationMs) { this.durationMs = durationMs; }
    public int getTensorCount() { return tensorCount; }
    public void setTensorCount(int tensorCount) { this.tensorCount = tensorCount; }
    public long getParamCount() { return paramCount; }
    public void setParamCount(long paramCount) { this.paramCount = paramCount; }
    public double getAvgQuantErrorMse() { return avgQuantErrorMse; }
    public void setAvgQuantErrorMse(double avgQuantErrorMse) { this.avgQuantErrorMse = avgQuantErrorMse; }
    public String getErrorMessage() { return errorMessage; }
    public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
    public String getStrategy() { return strategy; }
    public void setStrategy(String strategy) { this.strategy = strategy; }
}
