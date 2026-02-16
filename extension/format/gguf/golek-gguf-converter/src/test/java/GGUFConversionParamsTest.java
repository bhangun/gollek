import org.junit.jupiter.api.Test;

import tech.kayys.golek.converter.model.GGUFConversionParams;
import tech.kayys.golek.converter.model.QuantizationType;

import org.junit.jupiter.api.DisplayName;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for GGUF conversion parameters.
 *
 * @author Bhangun
 */
class GGUFConversionParamsTest {

    @Test
    @DisplayName("Should create valid conversion parameters")
    void testValidParamsCreation() {
        Path input = Path.of("/input/model.bin");
        Path output = Path.of("/output/model.gguf");

        GGUFConversionParams params = GGUFConversionParams.builder()
                .inputPath(input)
                .outputPath(output)
                .quantization(QuantizationType.Q4_K_M)
                .build();

        assertEquals(input, params.getInputPath());
        assertEquals(output, params.getOutputPath());
        assertEquals(QuantizationType.Q4_K_M, params.getQuantization());
        assertFalse(params.isVocabOnly());
        assertTrue(params.isUseMmap());
        assertEquals(0, params.getNumThreads());
        assertEquals(0, params.getPadVocab());
        assertNotNull(params.getMetadata());
        assertTrue(params.getMetadata().isEmpty());
    }

    @Test
    @DisplayName("Should use default quantization when not specified")
    void testDefaultQuantization() {
        Path input = Path.of("/input/model.bin");
        Path output = Path.of("/output/model.gguf");

        GGUFConversionParams params = GGUFConversionParams.builder()
                .inputPath(input)
                .outputPath(output)
                .build();

        assertEquals(QuantizationType.F16, params.getQuantization());
    }

    @Test
    @DisplayName("Should validate parameters correctly")
    void testParameterValidation() {
        Path input = Path.of("/input/model.bin");
        Path output = Path.of("/output/model.gguf");

        // Valid parameters
        GGUFConversionParams validParams = GGUFConversionParams.builder()
                .inputPath(input)
                .outputPath(output)
                .quantization(QuantizationType.Q4_K_M)
                .build();

        assertDoesNotThrow(() -> validParams.validate());

        // Invalid: null input
        assertThrows(IllegalArgumentException.class, () -> {
            GGUFConversionParams.builder()
                    .inputPath(null)
                    .outputPath(output)
                    .quantization(QuantizationType.Q4_K_M)
                    .build()
                    .validate();
        });

        // Invalid: null output
        assertThrows(IllegalArgumentException.class, () -> {
            GGUFConversionParams.builder()
                    .inputPath(input)
                    .outputPath(null)
                    .quantization(QuantizationType.Q4_K_M)
                    .build()
                    .validate();
        });

        // Invalid: null quantization
        assertThrows(IllegalArgumentException.class, () -> {
            GGUFConversionParams.builder()
                    .inputPath(input)
                    .outputPath(output)
                    .quantization(null)
                    .build()
                    .validate();
        });

        // Invalid: negative threads
        assertThrows(IllegalArgumentException.class, () -> {
            GGUFConversionParams.builder()
                    .inputPath(input)
                    .outputPath(output)
                    .quantization(QuantizationType.Q4_K_M)
                    .numThreads(-1)
                    .build()
                    .validate();
        });

        // Invalid: negative pad vocab
        assertThrows(IllegalArgumentException.class, () -> {
            GGUFConversionParams.builder()
                    .inputPath(input)
                    .outputPath(output)
                    .quantization(QuantizationType.Q4_K_M)
                    .padVocab(-1)
                    .build()
                    .validate();
        });
    }

    @Test
    @DisplayName("Should create builder with custom values")
    void testBuilderWithCustomValues() {
        Path input = Path.of("/custom/input.bin");
        Path output = Path.of("/custom/output.gguf");

        GGUFConversionParams params = GGUFConversionParams.builder()
                .inputPath(input)
                .outputPath(output)
                .modelType("llama")
                .quantization(QuantizationType.Q8_0)
                .vocabOnly(true)
                .useMmap(false)
                .numThreads(4)
                .vocabType("bpe")
                .padVocab(1024)
                .metadata(Map.of("author", "test", "version", "1.0"))
                .build();

        assertEquals(input, params.getInputPath());
        assertEquals(output, params.getOutputPath());
        assertEquals("llama", params.getModelType());
        assertEquals(QuantizationType.Q8_0, params.getQuantization());
        assertTrue(params.isVocabOnly());
        assertFalse(params.isUseMmap());
        assertEquals(4, params.getNumThreads());
        assertEquals("bpe", params.getVocabType());
        assertEquals(1024, params.getPadVocab());
        assertEquals(2, params.getMetadata().size());
        assertEquals("test", params.getMetadata().get("author"));
        assertEquals("1.0", params.getMetadata().get("version"));
    }

    @Test
    @DisplayName("Should create copy with builder")
    void testToBuilder() {
        Path input = Path.of("/original/input.bin");
        Path output = Path.of("/original/output.gguf");

        GGUFConversionParams original = GGUFConversionParams.builder()
                .inputPath(input)
                .outputPath(output)
                .modelType("mistral")
                .quantization(QuantizationType.Q4_K_S)
                .vocabOnly(true)
                .useMmap(false)
                .numThreads(8)
                .vocabType("spm")
                .padVocab(512)
                .metadata(Map.of("source", "huggingface"))
                .build();

        GGUFConversionParams copy = original.toBuilder()
                .modelType("llama") // Change model type
                .numThreads(4) // Change thread count
                .build();

        assertEquals("llama", copy.getModelType()); // Changed
        assertEquals(4, copy.getNumThreads()); // Changed
        assertEquals(QuantizationType.Q4_K_S, copy.getQuantization()); // Unchanged
        assertEquals("huggingface", copy.getMetadata().get("source")); // Unchanged
        assertTrue(copy.isVocabOnly()); // Unchanged
        assertFalse(copy.isUseMmap()); // Unchanged
        assertEquals("spm", copy.getVocabType()); // Unchanged
        assertEquals(512, copy.getPadVocab()); // Unchanged
    }
}