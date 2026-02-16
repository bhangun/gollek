package tech.kayys.golek.converter.model;

import java.nio.file.Path;
import java.util.Map;
import java.util.HashMap;

/**
 * Conversion parameters for GGUF model conversion.
 *
 * <p>
 * This class provides a type-safe, builder-based API for configuring
 * model conversions with sensible defaults.
 *
 * @author Bhangun
 * @version 1.0.0
 */
public class GGUFConversionParams {

    /**
     * Input model path (file or directory).
     * Required.
     */
    private final Path inputPath;

    /**
     * Output GGUF file path.
     * Required.
     */
    private final Path outputPath;

    /**
     * Model architecture hint (e.g., "llama", "mistral", "phi").
     * Optional - will be auto-detected if not provided.
     */
    private final String modelType;

    /**
     * Quantization type.
     * Default: f16
     *
     * Available types:
     * - f32, f16 (no quantization)
     * - q4_0, q4_1, q5_0, q5_1, q8_0, q8_1
     * - q2_k, q3_k_s, q3_k_m, q3_k_l
     * - q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k
     */
    private final QuantizationType quantization;

    /**
     * Convert vocabulary only (skip weights).
     * Default: false
     */
    private final boolean vocabOnly;

    /**
     * Use memory mapping for large files.
     * Default: true
     */
    private final boolean useMmap;

    /**
     * Number of threads for conversion (0 = auto).
     * Default: 0 (auto-detect)
     */
    private final int numThreads;

    /**
     * Vocabulary type override.
     * Optional - will be auto-detected if not provided.
     * Valid values: "bpe", "spm"
     */
    private final String vocabType;

    /**
     * Pad vocabulary to multiple of this value.
     * Default: 0 (no padding)
     */
    private final int padVocab;

    /**
     * Additional metadata key-value pairs to include in the GGUF file.
     */
    private final Map<String, String> metadata;

    // Constructor for builder
    private GGUFConversionParams(Builder builder) {
        this.inputPath = builder.inputPath;
        this.outputPath = builder.outputPath;
        this.modelType = builder.modelType;
        this.quantization = builder.quantization != null ? builder.quantization : QuantizationType.F16;
        this.vocabOnly = builder.vocabOnly;
        this.useMmap = builder.useMmap;
        this.numThreads = builder.numThreads;
        this.vocabType = builder.vocabType;
        this.padVocab = builder.padVocab;
        this.metadata = builder.metadata != null ? new HashMap<>(builder.metadata) : new HashMap<>();
        
        validate();
    }

    /**
     * Validate parameters.
     *
     * @throws IllegalArgumentException if parameters are invalid
     */
    public void validate() {
        if (inputPath == null) {
            throw new IllegalArgumentException("Input path is required");
        }
        if (outputPath == null) {
            throw new IllegalArgumentException("Output path is required");
        }
        if (quantization == null) {
            throw new IllegalArgumentException("Quantization type is required");
        }
        if (numThreads < 0) {
            throw new IllegalArgumentException("Number of threads must be >= 0");
        }
        if (padVocab < 0) {
            throw new IllegalArgumentException("Pad vocab must be >= 0");
        }
    }

    public Path getInputPath() {
        return inputPath;
    }

    public Path getOutputPath() {
        return outputPath;
    }

    public String getModelType() {
        return modelType;
    }

    public QuantizationType getQuantization() {
        return quantization;
    }

    public boolean isVocabOnly() {
        return vocabOnly;
    }

    public boolean isUseMmap() {
        return useMmap;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public String getVocabType() {
        return vocabType;
    }

    public int getPadVocab() {
        return padVocab;
    }

    public Map<String, String> getMetadata() {
        return new HashMap<>(metadata);
    }

    /**
     * Create a copy with modified parameters.
     *
     * @return new builder initialized with current values
     */
    public Builder toBuilder() {
        return new Builder()
                .inputPath(inputPath)
                .outputPath(outputPath)
                .modelType(modelType)
                .quantization(quantization)
                .vocabOnly(vocabOnly)
                .useMmap(useMmap)
                .numThreads(numThreads)
                .vocabType(vocabType)
                .padVocab(padVocab)
                .metadata(new HashMap<>(metadata));
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder class for GGUFConversionParams.
     */
    public static class Builder {
        private Path inputPath;
        private Path outputPath;
        private String modelType;
        private QuantizationType quantization;
        private boolean vocabOnly = false;
        private boolean useMmap = true;
        private int numThreads = 0;
        private String vocabType;
        private int padVocab = 0;
        private Map<String, String> metadata;

        public Builder inputPath(Path inputPath) {
            this.inputPath = inputPath;
            return this;
        }

        public Builder outputPath(Path outputPath) {
            this.outputPath = outputPath;
            return this;
        }

        public Builder modelType(String modelType) {
            this.modelType = modelType;
            return this;
        }

        public Builder quantization(QuantizationType quantization) {
            this.quantization = quantization;
            return this;
        }

        public Builder vocabOnly(boolean vocabOnly) {
            this.vocabOnly = vocabOnly;
            return this;
        }

        public Builder useMmap(boolean useMmap) {
            this.useMmap = useMmap;
            return this;
        }

        public Builder numThreads(int numThreads) {
            this.numThreads = numThreads;
            return this;
        }

        public Builder vocabType(String vocabType) {
            this.vocabType = vocabType;
            return this;
        }

        public Builder padVocab(int padVocab) {
            this.padVocab = padVocab;
            return this;
        }

        public Builder metadata(Map<String, String> metadata) {
            this.metadata = metadata;
            return this;
        }

        public GGUFConversionParams build() {
            return new GGUFConversionParams(this);
        }
    }
}
