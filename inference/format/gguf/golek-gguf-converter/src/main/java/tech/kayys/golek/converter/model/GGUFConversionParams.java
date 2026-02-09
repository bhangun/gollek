package tech.kayys.golek.converter.model;

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;

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
@Data
@Builder
public class GGUFConversionParams {

    /**
     * Input model path (file or directory).
     * Required.
     */
    @NonNull
    private final Path inputPath;

    /**
     * Output GGUF file path.
     * Required.
     */
    @NonNull
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
    @NonNull
    @Builder.Default
    private final QuantizationType quantization = QuantizationType.F16;

    /**
     * Convert vocabulary only (skip weights).
     * Default: false
     */
    @Builder.Default
    private final boolean vocabOnly = false;

    /**
     * Use memory mapping for large files.
     * Default: true
     */
    @Builder.Default
    private final boolean useMmap = true;

    /**
     * Number of threads for conversion (0 = auto).
     * Default: 0 (auto-detect)
     */
    @Builder.Default
    private final int numThreads = 0;

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
    @Builder.Default
    private final int padVocab = 0;

    /**
     * Additional metadata key-value pairs to include in the GGUF file.
     */
    @Builder.Default
    private final Map<String, String> metadata = new HashMap<>();

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

    /**
     * Create a copy with modified parameters.
     * 
     * @return new builder initialized with current values
     */
    public GGUFConversionParamsBuilder toBuilder() {
        return builder()
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
}
