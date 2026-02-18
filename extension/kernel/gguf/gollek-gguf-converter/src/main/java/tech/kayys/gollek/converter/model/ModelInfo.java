package tech.kayys.gollek.converter.model;

import tech.kayys.gollek.spi.model.ModelFormat;

/**
 * Model metadata and information.
 * 
 * <p>
 * Contains information about a model's architecture, parameters,
 * and resource requirements.
 * 
 * @author Bhangun
 * @version 1.0.0
 */
public class ModelInfo {

    /**
     * Model type/family (e.g., "llama", "mistral", "gpt2").
     */
    private final String modelType;

    /**
     * Architecture name (e.g., "LlamaForCausalLM", "GPT2LMHeadModel").
     */
    private final String architecture;

    /**
     * Total parameter count.
     */
    private final long parameterCount;

    /**
     * Number of transformer layers.
     */
    private final int numLayers;

    /**
     * Hidden dimension size.
     */
    private final int hiddenSize;

    /**
     * Vocabulary size.
     */
    private final int vocabSize;

    /**
     * Maximum context length.
     */
    private final int contextLength;

    /**
     * Current quantization (for GGUF files).
     */
    private final String quantization;

    /**
     * File size in bytes.
     */
    private final long fileSize;

    /**
     * Source format (e.g., "pytorch", "safetensors", "gguf").
     */
    private final ModelFormat format;

    public ModelInfo(String modelType, String architecture, long parameterCount, int numLayers, int hiddenSize,
            int vocabSize, int contextLength, String quantization, long fileSize, ModelFormat format) {
        this.modelType = modelType;
        this.architecture = architecture;
        this.parameterCount = parameterCount;
        this.numLayers = numLayers;
        this.hiddenSize = hiddenSize;
        this.vocabSize = vocabSize;
        this.contextLength = contextLength;
        this.quantization = quantization;
        this.fileSize = fileSize;
        this.format = format;
    }

    public String getModelType() {
        return modelType;
    }

    public String getArchitecture() {
        return architecture;
    }

    public long getParameterCount() {
        return parameterCount;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public int getVocabSize() {
        return vocabSize;
    }

    public int getContextLength() {
        return contextLength;
    }

    public String getQuantization() {
        return quantization;
    }

    public long getFileSize() {
        return fileSize;
    }

    public ModelFormat getFormat() {
        return format;
    }

    /**
     * Get file size in gigabytes.
     * 
     * @return size in GB
     */
    public double getFileSizeGb() {
        return fileSize / (1024.0 * 1024.0 * 1024.0);
    }

    /**
     * Get human-readable parameter count.
     * 
     * @return formatted string (e.g., "7.2B", "13B")
     */
    public String getParameterCountFormatted() {
        if (parameterCount == 0) {
            return "Unknown";
        }

        double billions = parameterCount / 1_000_000_000.0;
        if (billions >= 1.0) {
            return String.format("%.1fB", billions);
        }

        double millions = parameterCount / 1_000_000.0;
        return String.format("%.1fM", millions);
    }

    /**
     * Get human-readable file size.
     * 
     * @return formatted string (e.g., "4.2 GB", "512 MB")
     */
    public String getFileSizeFormatted() {
        if (fileSize == 0) {
            return "Unknown";
        }

        double gb = fileSize / (1024.0 * 1024.0 * 1024.0);
        if (gb >= 1.0) {
            return String.format("%.2f GB", gb);
        }

        double mb = fileSize / (1024.0 * 1024.0);
        return String.format("%.2f MB", mb);
    }

    /**
     * Check if this is a large model (>10B parameters).
     * 
     * @return true if large model
     */
    public boolean isLargeModel() {
        return parameterCount > 10_000_000_000L;
    }

    /**
     * Estimate memory requirements in GB for inference.
     * 
     * @param quantType quantization type (null for current/none)
     * @return estimated memory in GB
     */
    public double estimateMemoryGb(QuantizationType quantType) {
        if (parameterCount == 0) {
            // Fallback to file size estimate
            return getFileSizeGb() * 1.2; // Add 20% overhead
        }

        double bytesPerParam;
        if (quantType != null) {
            bytesPerParam = 4.0 / quantType.getCompressionRatio();
        } else if (quantization != null && !quantization.isEmpty()) {
            QuantizationType currentQuant = QuantizationType.fromNativeName(quantization);
            bytesPerParam = currentQuant != null ? 4.0 / currentQuant.getCompressionRatio() : 4.0;
        } else {
            bytesPerParam = 4.0; // Assume FP32
        }

        double modelSizeGb = (parameterCount * bytesPerParam) / (1024.0 * 1024.0 * 1024.0);

        // Add overhead for KV cache and activations
        double overhead = Math.max(2.0, modelSizeGb * 0.2);

        return modelSizeGb + overhead;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ModelInfo{");
        if (modelType != null) {
            sb.append("type=").append(modelType);
        }
        if (architecture != null) {
            sb.append(", arch=").append(architecture);
        }
        if (parameterCount > 0) {
            sb.append(", params=").append(getParameterCountFormatted());
        }
        if (fileSize > 0) {
            sb.append(", size=").append(getFileSizeFormatted());
        }
        if (format != null) {
            sb.append(", format=").append(format);
        }
        sb.append("}");
        return sb.toString();
    }

    /**
     * Builder class for ModelInfo.
     */
    public static class Builder {
        private String modelType;
        private String architecture;
        private long parameterCount;
        private int numLayers;
        private int hiddenSize;
        private int vocabSize;
        private int contextLength;
        private String quantization;
        private long fileSize;
        private ModelFormat format;

        public Builder modelType(String modelType) {
            this.modelType = modelType;
            return this;
        }

        public Builder architecture(String architecture) {
            this.architecture = architecture;
            return this;
        }

        public Builder parameterCount(long parameterCount) {
            this.parameterCount = parameterCount;
            return this;
        }

        public Builder numLayers(int numLayers) {
            this.numLayers = numLayers;
            return this;
        }

        public Builder hiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }

        public Builder vocabSize(int vocabSize) {
            this.vocabSize = vocabSize;
            return this;
        }

        public Builder contextLength(int contextLength) {
            this.contextLength = contextLength;
            return this;
        }

        public Builder quantization(String quantization) {
            this.quantization = quantization;
            return this;
        }

        public Builder fileSize(long fileSize) {
            this.fileSize = fileSize;
            return this;
        }

        public Builder format(ModelFormat format) {
            this.format = format;
            return this;
        }

        public ModelInfo build() {
            return new ModelInfo(modelType, architecture, parameterCount, numLayers, hiddenSize, vocabSize, contextLength, quantization, fileSize, format);
        }
    }

    public static Builder builder() {
        return new Builder();
    }
}