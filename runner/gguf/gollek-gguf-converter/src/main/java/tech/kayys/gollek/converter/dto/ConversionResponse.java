package tech.kayys.gollek.converter.dto;

import com.fasterxml.jackson.annotation.JsonInclude;

import tech.kayys.gollek.converter.model.ConversionResult;

/**
 * Response DTO for conversion operations.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ConversionResponse {

    private long conversionId;
    private boolean success;
    private String requestId;
    private String inputPath;
    private String outputPath;
    private String outputSize;
    private String duration;
    private double compressionRatio;
    private ModelInfoResponse inputInfo;
    private String errorMessage;
    private boolean dryRun;
    private String derivedOutputName;
    private String inputBasePath;
    private String outputBasePath;

    public ConversionResponse() {
    }

    public ConversionResponse(long conversionId, boolean success, String requestId, String inputPath, String outputPath,
            String outputSize, String duration, double compressionRatio, ModelInfoResponse inputInfo,
            String errorMessage, boolean dryRun, String derivedOutputName, String inputBasePath, String outputBasePath) {
        this.conversionId = conversionId;
        this.success = success;
        this.requestId = requestId;
        this.inputPath = inputPath;
        this.outputPath = outputPath;
        this.outputSize = outputSize;
        this.duration = duration;
        this.compressionRatio = compressionRatio;
        this.inputInfo = inputInfo;
        this.errorMessage = errorMessage;
        this.dryRun = dryRun;
        this.derivedOutputName = derivedOutputName;
        this.inputBasePath = inputBasePath;
        this.outputBasePath = outputBasePath;
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

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
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

    public boolean isDryRun() {
        return dryRun;
    }

    public void setDryRun(boolean dryRun) {
        this.dryRun = dryRun;
    }

    public String getDerivedOutputName() {
        return derivedOutputName;
    }

    public void setDerivedOutputName(String derivedOutputName) {
        this.derivedOutputName = derivedOutputName;
    }

    public String getInputBasePath() {
        return inputBasePath;
    }

    public void setInputBasePath(String inputBasePath) {
        this.inputBasePath = inputBasePath;
    }

    public String getOutputBasePath() {
        return outputBasePath;
    }

    public void setOutputBasePath(String outputBasePath) {
        this.outputBasePath = outputBasePath;
    }

    public static ConversionResponse fromResult(ConversionResult result, String requestId) {
        return ConversionResponse.builder()
                .conversionId(result.getConversionId())
                .success(result.isSuccess())
                .requestId(requestId)
                .outputPath(result.getOutputPath().toString())
                .outputSize(result.getOutputSizeFormatted())
                .duration(result.getDurationFormatted())
                .compressionRatio(result.getCompressionRatio())
                .inputInfo(ModelInfoResponse.fromModelInfo(result.getInputInfo()))
                .errorMessage(result.getErrorMessage())
                .dryRun(false)
                .build();
    }

    public static ConversionResponseBuilder builder() {
        return new ConversionResponseBuilder();
    }

    public static class ConversionResponseBuilder {
        private long conversionId;
        private boolean success;
        private String requestId;
        private String inputPath;
        private String outputPath;
        private String outputSize;
        private String duration;
        private double compressionRatio;
        private ModelInfoResponse inputInfo;
        private String errorMessage;
        private boolean dryRun;
        private String derivedOutputName;
        private String inputBasePath;
        private String outputBasePath;

        public ConversionResponseBuilder conversionId(long conversionId) {
            this.conversionId = conversionId;
            return this;
        }

        public ConversionResponseBuilder success(boolean success) {
            this.success = success;
            return this;
        }

        public ConversionResponseBuilder requestId(String requestId) {
            this.requestId = requestId;
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

        public ConversionResponseBuilder dryRun(boolean dryRun) {
            this.dryRun = dryRun;
            return this;
        }

        public ConversionResponseBuilder derivedOutputName(String derivedOutputName) {
            this.derivedOutputName = derivedOutputName;
            return this;
        }

        public ConversionResponseBuilder inputBasePath(String inputBasePath) {
            this.inputBasePath = inputBasePath;
            return this;
        }

        public ConversionResponseBuilder outputBasePath(String outputBasePath) {
            this.outputBasePath = outputBasePath;
            return this;
        }

        public ConversionResponse build() {
            return new ConversionResponse(conversionId, success, requestId, inputPath, outputPath, outputSize,
                    duration, compressionRatio, inputInfo, errorMessage, dryRun,
                    derivedOutputName, inputBasePath, outputBasePath);
        }
    }
}
