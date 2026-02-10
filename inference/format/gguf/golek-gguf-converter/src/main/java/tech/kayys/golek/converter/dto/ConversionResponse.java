package tech.kayys.golek.converter.dto;

import com.fasterxml.jackson.annotation.JsonInclude;

import tech.kayys.golek.converter.model.ConversionResult;

/**
 * Response DTO for conversion operations.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ConversionResponse {

    private long conversionId;
    private boolean success;
    private String tenantId;
    private String inputPath;
    private String outputPath;
    private String outputSize;
    private String duration;
    private double compressionRatio;
    private ModelInfoResponse inputInfo;
    private String errorMessage;

    public ConversionResponse() {
    }

    public ConversionResponse(long conversionId, boolean success, String tenantId, String inputPath, String outputPath,
            String outputSize, String duration, double compressionRatio, ModelInfoResponse inputInfo,
            String errorMessage) {
        this.conversionId = conversionId;
        this.success = success;
        this.tenantId = tenantId;
        this.inputPath = inputPath;
        this.outputPath = outputPath;
        this.outputSize = outputSize;
        this.duration = duration;
        this.compressionRatio = compressionRatio;
        this.inputInfo = inputInfo;
        this.errorMessage = errorMessage;
    }

    public long getConversionId() {
        return conversionId;
    }

    public void setConversionId(long conversionId) {
        this.conversionId = conversionId;
    }

    public boolean isSuccess() {
        return success;
    }

    public void setSuccess(boolean success) {
        this.success = success;
    }

    public String getTenantId() {
        return tenantId;
    }

    public void setTenantId(String tenantId) {
        this.tenantId = tenantId;
    }

    public String getInputPath() {
        return inputPath;
    }

    public void setInputPath(String inputPath) {
        this.inputPath = inputPath;
    }

    public String getOutputPath() {
        return outputPath;
    }

    public void setOutputPath(String outputPath) {
        this.outputPath = outputPath;
    }

    public String getOutputSize() {
        return outputSize;
    }

    public void setOutputSize(String outputSize) {
        this.outputSize = outputSize;
    }

    public String getDuration() {
        return duration;
    }

    public void setDuration(String duration) {
        this.duration = duration;
    }

    public double getCompressionRatio() {
        return compressionRatio;
    }

    public void setCompressionRatio(double compressionRatio) {
        this.compressionRatio = compressionRatio;
    }

    public ModelInfoResponse getInputInfo() {
        return inputInfo;
    }

    public void setInputInfo(ModelInfoResponse inputInfo) {
        this.inputInfo = inputInfo;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    static ConversionResponse fromResult(ConversionResult result, String tenantId) {
        return ConversionResponse.builder()
                .conversionId(result.getConversionId())
                .success(result.isSuccess())
                .tenantId(tenantId)
                .outputPath(result.getOutputPath().toString())
                .outputSize(result.getOutputSizeFormatted())
                .duration(result.getDurationFormatted())
                .compressionRatio(result.getCompressionRatio())
                .inputInfo(ModelInfoResponse.fromModelInfo(result.getInputInfo()))
                .errorMessage(result.getErrorMessage())
                .build();
    }

    public static ConversionResponseBuilder builder() {
        return new ConversionResponseBuilder();
    }

    public static class ConversionResponseBuilder {
        private long conversionId;
        private boolean success;
        private String tenantId;
        private String inputPath;
        private String outputPath;
        private String outputSize;
        private String duration;
        private double compressionRatio;
        private ModelInfoResponse inputInfo;
        private String errorMessage;

        public ConversionResponseBuilder conversionId(long conversionId) {
            this.conversionId = conversionId;
            return this;
        }

        public ConversionResponseBuilder success(boolean success) {
            this.success = success;
            return this;
        }

        public ConversionResponseBuilder tenantId(String tenantId) {
            this.tenantId = tenantId;
            return this;
        }

        public ConversionResponseBuilder inputPath(String inputPath) {
            this.inputPath = inputPath;
            return this;
        }

        public ConversionResponseBuilder outputPath(String outputPath) {
            this.outputPath = outputPath;
            return this;
        }

        public ConversionResponseBuilder outputSize(String outputSize) {
            this.outputSize = outputSize;
            return this;
        }

        public ConversionResponseBuilder duration(String duration) {
            this.duration = duration;
            return this;
        }

        public ConversionResponseBuilder compressionRatio(double compressionRatio) {
            this.compressionRatio = compressionRatio;
            return this;
        }

        public ConversionResponseBuilder inputInfo(ModelInfoResponse inputInfo) {
            this.inputInfo = inputInfo;
            return this;
        }

        public ConversionResponseBuilder errorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
            return this;
        }

        public ConversionResponse build() {
            return new ConversionResponse(conversionId, success, tenantId, inputPath, outputPath, outputSize,
                    duration, compressionRatio, inputInfo, errorMessage);
        }
    }
}