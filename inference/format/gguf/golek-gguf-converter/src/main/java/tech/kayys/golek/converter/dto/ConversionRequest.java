package tech.kayys.golek.converter.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import tech.kayys.golek.converter.model.QuantizationType;

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
}
