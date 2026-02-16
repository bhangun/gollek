package tech.kayys.golek.converter.dto;

import com.fasterxml.jackson.annotation.JsonInclude;

import tech.kayys.golek.converter.model.ModelInfo;

/**
 * Model information response DTO.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ModelInfoResponse {

    private String modelType;
    private String architecture;
    private String parameterCount;
    private long parameterCountRaw;
    private int numLayers;
    private int hiddenSize;
    private int vocabSize;
    private int contextLength;
    private String quantization;
    private String fileSize;
    private long fileSizeBytes;
    private String format;
    private double estimatedMemoryGb;

    public ModelInfoResponse() {
    }

    public ModelInfoResponse(String modelType, String architecture, String parameterCount, long parameterCountRaw,
            int numLayers, int hiddenSize, int vocabSize, int contextLength, String quantization, String fileSize,
            long fileSizeBytes, String format, double estimatedMemoryGb) {
        this.modelType = modelType;
        this.architecture = architecture;
        this.parameterCount = parameterCount;
        this.parameterCountRaw = parameterCountRaw;
        this.numLayers = numLayers;
        this.hiddenSize = hiddenSize;
        this.vocabSize = vocabSize;
        this.contextLength = contextLength;
        this.quantization = quantization;
        this.fileSize = fileSize;
        this.fileSizeBytes = fileSizeBytes;
        this.format = format;
        this.estimatedMemoryGb = estimatedMemoryGb;
    }

    public String getModelType() {
        return modelType;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public String getArchitecture() {
        return architecture;
    }

    public void setArchitecture(String architecture) {
        this.architecture = architecture;
    }

    public String getParameterCount() {
        return parameterCount;
    }

    public void setParameterCount(String parameterCount) {
        this.parameterCount = parameterCount;
    }

    public long getParameterCountRaw() {
        return parameterCountRaw;
    }

    public void setParameterCountRaw(long parameterCountRaw) {
        this.parameterCountRaw = parameterCountRaw;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public void setNumLayers(int numLayers) {
        this.numLayers = numLayers;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
    }

    public int getVocabSize() {
        return vocabSize;
    }

    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }

    public int getContextLength() {
        return contextLength;
    }

    public void setContextLength(int contextLength) {
        this.contextLength = contextLength;
    }

    public String getQuantization() {
        return quantization;
    }

    public void setQuantization(String quantization) {
        this.quantization = quantization;
    }

    public String getFileSize() {
        return fileSize;
    }

    public void setFileSize(String fileSize) {
        this.fileSize = fileSize;
    }

    public long getFileSizeBytes() {
        return fileSizeBytes;
    }

    public void setFileSizeBytes(long fileSizeBytes) {
        this.fileSizeBytes = fileSizeBytes;
    }

    public String getFormat() {
        return format;
    }

    public void setFormat(String format) {
        this.format = format;
    }

    public double getEstimatedMemoryGb() {
        return estimatedMemoryGb;
    }

    public void setEstimatedMemoryGb(double estimatedMemoryGb) {
        this.estimatedMemoryGb = estimatedMemoryGb;
    }

    public static ModelInfoResponse fromModelInfo(ModelInfo info) {
        if (info == null) {
            return null;
        }

        return ModelInfoResponse.builder()
                .modelType(info.getModelType())
                .architecture(info.getArchitecture())
                .parameterCount(info.getParameterCountFormatted())
                .parameterCountRaw(info.getParameterCount())
                .numLayers(info.getNumLayers())
                .hiddenSize(info.getHiddenSize())
                .vocabSize(info.getVocabSize())
                .contextLength(info.getContextLength())
                .quantization(info.getQuantization())
                .fileSize(info.getFileSizeFormatted())
                .fileSizeBytes(info.getFileSize())
                .format(info.getFormat() != null ? info.getFormat().getDisplayName() : null)
                .estimatedMemoryGb(info.estimateMemoryGb(null))
                .build();
    }

    public static ModelInfoResponseBuilder builder() {
        return new ModelInfoResponseBuilder();
    }

    public static class ModelInfoResponseBuilder {
        private String modelType;
        private String architecture;
        private String parameterCount;
        private long parameterCountRaw;
        private int numLayers;
        private int hiddenSize;
        private int vocabSize;
        private int contextLength;
        private String quantization;
        private String fileSize;
        private long fileSizeBytes;
        private String format;
        private double estimatedMemoryGb;

        public ModelInfoResponseBuilder modelType(String modelType) {
            this.modelType = modelType;
            return this;
        }

        public ModelInfoResponseBuilder architecture(String architecture) {
            this.architecture = architecture;
            return this;
        }

        public ModelInfoResponseBuilder parameterCount(String parameterCount) {
            this.parameterCount = parameterCount;
            return this;
        }

        public ModelInfoResponseBuilder parameterCountRaw(long parameterCountRaw) {
            this.parameterCountRaw = parameterCountRaw;
            return this;
        }

        public ModelInfoResponseBuilder numLayers(int numLayers) {
            this.numLayers = numLayers;
            return this;
        }

        public ModelInfoResponseBuilder hiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }

        public ModelInfoResponseBuilder vocabSize(int vocabSize) {
            this.vocabSize = vocabSize;
            return this;
        }

        public ModelInfoResponseBuilder contextLength(int contextLength) {
            this.contextLength = contextLength;
            return this;
        }

        public ModelInfoResponseBuilder quantization(String quantization) {
            this.quantization = quantization;
            return this;
        }

        public ModelInfoResponseBuilder fileSize(String fileSize) {
            this.fileSize = fileSize;
            return this;
        }

        public ModelInfoResponseBuilder fileSizeBytes(long fileSizeBytes) {
            this.fileSizeBytes = fileSizeBytes;
            return this;
        }

        public ModelInfoResponseBuilder format(String format) {
            this.format = format;
            return this;
        }

        public ModelInfoResponseBuilder estimatedMemoryGb(double estimatedMemoryGb) {
            this.estimatedMemoryGb = estimatedMemoryGb;
            return this;
        }

        public ModelInfoResponse build() {
            return new ModelInfoResponse(modelType, architecture, parameterCount, parameterCountRaw, numLayers,
                    hiddenSize,
                    vocabSize, contextLength, quantization, fileSize, fileSizeBytes, format, estimatedMemoryGb);
        }
    }
}
