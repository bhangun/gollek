package tech.kayys.gollek.converter.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import tech.kayys.gollek.converter.model.QuantizationType;

/**
 * Request DTOs for GGUF converter API.
 * 
 * @author Bhangun
 * @version 1.0.0
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ConversionRequest {

    /**
     * Input model path (relative to tenant storage).
     */
    @NotBlank(message = "Input path is required")
    private String inputPath;

    /**
     * Output GGUF file path (relative to tenant storage).
     */
    @NotBlank(message = "Output path is required")
    private String outputPath;

    /**
     * Model type hint (optional, will be auto-detected).
     */
    private String modelType;

    /**
     * Quantization type.
     */
    @NotNull(message = "Quantization type is required")
    private QuantizationType quantization = QuantizationType.F16;

    /**
     * Convert vocabulary only.
     */
    private boolean vocabOnly = false;

    /**
     * Number of threads (0 = auto).
     */
    private int numThreads = 0;

    /**
     * Vocabulary type override.
     */
    private String vocabType;

    public ConversionRequest() {
    }

    public ConversionRequest(String inputPath, String outputPath, String modelType, QuantizationType quantization,
            boolean vocabOnly, int numThreads, String vocabType) {
        this.inputPath = inputPath;
        this.outputPath = outputPath;
        this.modelType = modelType;
        this.quantization = quantization;
        this.vocabOnly = vocabOnly;
        this.numThreads = numThreads;
        this.vocabType = vocabType;
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

    public String getModelType() {
        return modelType;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public QuantizationType getQuantization() {
        return quantization;
    }

    public void setQuantization(QuantizationType quantization) {
        this.quantization = quantization;
    }

    public boolean isVocabOnly() {
        return vocabOnly;
    }

    public void setVocabOnly(boolean vocabOnly) {
        this.vocabOnly = vocabOnly;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }

    public String getVocabType() {
        return vocabType;
    }

    public void setVocabType(String vocabType) {
        this.vocabType = vocabType;
    }

    public static ConversionRequestBuilder builder() {
        return new ConversionRequestBuilder();
    }

    public static class ConversionRequestBuilder {
        private String inputPath;
        private String outputPath;
        private String modelType;
        private QuantizationType quantization;
        private boolean vocabOnly;
        private int numThreads;
        private String vocabType;

        public ConversionRequestBuilder inputPath(String inputPath) {
            this.inputPath = inputPath;
            return this;
        }

        public ConversionRequestBuilder outputPath(String outputPath) {
            this.outputPath = outputPath;
            return this;
        }

        public ConversionRequestBuilder modelType(String modelType) {
            this.modelType = modelType;
            return this;
        }

        public ConversionRequestBuilder quantization(QuantizationType quantization) {
            this.quantization = quantization;
            return this;
        }

        public ConversionRequestBuilder vocabOnly(boolean vocabOnly) {
            this.vocabOnly = vocabOnly;
            return this;
        }

        public ConversionRequestBuilder numThreads(int numThreads) {
            this.numThreads = numThreads;
            return this;
        }

        public ConversionRequestBuilder vocabType(String vocabType) {
            this.vocabType = vocabType;
            return this;
        }

        public ConversionRequest build() {
            return new ConversionRequest(inputPath, outputPath, modelType, quantization, vocabOnly, numThreads,
                    vocabType);
        }
    }
}
