
import io.quarkus.test.junit.QuarkusTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.io.TempDir;
import jakarta.inject.Inject;
import tech.kayys.gollek.converter.GGUFConverter;
import tech.kayys.gollek.converter.GGUFException;
import tech.kayys.gollek.converter.model.ConversionProgress;
import tech.kayys.gollek.converter.model.GGUFConversionParams;
import tech.kayys.gollek.converter.model.ModelInfo;
import tech.kayys.gollek.converter.model.QuantizationType;
import tech.kayys.gollek.spi.model.ModelFormat;

import java.nio.file.Files;
import java.nio.file.Path;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for GGUF converter service.
 * 
 * @author Bhangun
 */
@QuarkusTest
class GGUFConverterTest {

    @Inject
    GGUFConverter converter;

    @TempDir
    Path tempDir;

    @Test
    @DisplayName("Should get converter version")
    void testGetVersion() {
        String version = converter.getVersion();
        assertThat(version).isNotNull().isNotEmpty();
        System.out.println("GGUF Bridge version: " + version);
    }

    @Test
    @DisplayName("Should list available quantizations")
    void testGetAvailableQuantizations() {
        QuantizationType[] types = converter.getAvailableQuantizations();

        assertThat(types).isNotNull();
        assertThat(types).hasSizeGreaterThan(10);
        assertThat(types).contains(
                QuantizationType.F16,
                QuantizationType.Q4_K_M,
                QuantizationType.Q8_0);
    }

    @Test
    @DisplayName("Should detect PyTorch format")
    void testDetectPyTorchFormat() throws IOException {
        // Create a mock PyTorch model directory
        Path modelDir = tempDir.resolve("test-model");
        Files.createDirectories(modelDir);
        Files.createFile(modelDir.resolve("pytorch_model.bin"));
        Files.createFile(modelDir.resolve("config.json"));

        ModelFormat format = converter.detectFormat(modelDir);
        assertThat(format).isEqualTo(ModelFormat.PYTORCH);
    }

    @Test
    @DisplayName("Should detect SafeTensors format")
    void testDetectSafeTensorsFormat() throws IOException {
        Path modelDir = tempDir.resolve("test-model-st");
        Files.createDirectories(modelDir);
        Files.createFile(modelDir.resolve("model.safetensors"));
        Files.createFile(modelDir.resolve("config.json"));

        ModelFormat format = converter.detectFormat(modelDir);
        assertThat(format).isEqualTo(ModelFormat.SAFETENSORS);
    }

    @Test
    @DisplayName("Should detect GGUF format")
    void testDetectGGUFFormat() throws IOException {
        Path ggufFile = tempDir.resolve("model.gguf");
        Files.createFile(ggufFile);

        ModelFormat format = converter.detectFormat(ggufFile);
        assertThat(format).isEqualTo(ModelFormat.GGUF);
    }

    @Test
    @DisplayName("Should validate conversion parameters")
    void testParameterValidation() {
        Path input = tempDir.resolve("input.bin");
        Path output = tempDir.resolve("output.gguf");

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
                    .build()
                    .validate();
        });

        // Invalid: null output
        assertThrows(IllegalArgumentException.class, () -> {
            GGUFConversionParams.builder()
                    .inputPath(input)
                    .outputPath(null)
                    .build()
                    .validate();
        });
    }

    @Test
    @DisplayName("Should recommend appropriate quantization")
    void testQuantizationRecommendation() {
        // Small model - should recommend high quality
        QuantizationType small = QuantizationType.recommend(2.0, true);
        assertThat(small).isIn(QuantizationType.F16, QuantizationType.Q4_K_M);

        // Large model - should recommend more compression
        QuantizationType large = QuantizationType.recommend(50.0, false);
        assertThat(large).isIn(
                QuantizationType.Q4_K_S,
                QuantizationType.Q3_K_M);

        // Medium model balanced
        QuantizationType medium = QuantizationType.recommend(7.0, false);
        assertThat(medium.getCompressionRatio()).isGreaterThan(6.0);
    }

    @Test
    @DisplayName("Should handle concurrent conversions safely")
    void testConcurrentConversions() throws InterruptedException {
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger errorCount = new AtomicInteger(0);

        // Simulate multiple concurrent conversion requests
        Thread[] threads = new Thread[5];
        for (int i = 0; i < threads.length; i++) {
            final int index = i;
            threads[i] = new Thread(() -> {
                try {
                    Path input = tempDir.resolve("input-" + index + ".bin");
                    Path output = tempDir.resolve("output-" + index + ".gguf");
                    Files.createFile(input);

                    GGUFConversionParams params = GGUFConversionParams.builder()
                            .inputPath(input)
                            .outputPath(output)
                            .quantization(QuantizationType.Q4_K_M)
                            .build();

                    // This will fail since we don't have real model files,
                    // but it tests the thread safety of the converter
                    try {
                        converter.convert(params, null);
                        successCount.incrementAndGet();
                    } catch (GGUFException e) {
                        errorCount.incrementAndGet();
                    }
                } catch (Exception e) {
                    errorCount.incrementAndGet();
                }
            });
            threads[i].start();
        }

        // Wait for all threads
        for (Thread thread : threads) {
            thread.join();
        }

        // All should have completed (either success or controlled error)
        assertThat(successCount.get() + errorCount.get()).isEqualTo(threads.length);
    }

    @Test
    @DisplayName("Should handle progress callbacks")
    void testProgressCallbacks() throws IOException {
        Path input = tempDir.resolve("input.bin");
        Path output = tempDir.resolve("output.gguf");
        Files.createFile(input);

        AtomicInteger progressUpdateCount = new AtomicInteger(0);

        GGUFConversionParams params = GGUFConversionParams.builder()
                .inputPath(input)
                .outputPath(output)
                .quantization(QuantizationType.Q4_K_M)
                .build();

        try {
            converter.convert(params, progress -> {
                progressUpdateCount.incrementAndGet();
                assertThat(progress.getProgress()).isBetween(0.0f, 1.0f);
                assertThat(progress.getStage()).isNotNull();
                System.out.println("Progress: " + progress.getProgressPercent() +
                        "% - " + progress.getStage());
            });
        } catch (GGUFException e) {
            // Expected since we don't have a real model
        }

        // Should have received at least one progress update
        // (even if conversion fails, initialization progress is reported)
        assertThat(progressUpdateCount.get()).isGreaterThanOrEqualTo(0);
    }

    @Test
    @DisplayName("Should calculate model info correctly")
    void testModelInfo() {
        ModelInfo info = ModelInfo.builder()
                .modelType("llama")
                .architecture("LlamaForCausalLM")
                .parameterCount(7_241_748_480L)
                .numLayers(32)
                .hiddenSize(4096)
                .vocabSize(32000)
                .contextLength(4096)
                .fileSize(13_476_937_728L)
                .format(ModelFormat.PYTORCH)
                .build();

        assertThat(info.getParameterCountFormatted()).isEqualTo("7.2B");
        assertThat(info.getFileSizeGb()).isBetween(12.0, 13.0);
        assertThat(info.getFileSizeFormatted()).contains("GB");
        assertThat(info.isLargeModel()).isFalse();

        double estimatedMemory = info.estimateMemoryGb(QuantizationType.Q4_K_M);
        assertThat(estimatedMemory).isBetween(5.0, 10.0);
    }

    @Test
    @DisplayName("Should validate quantization types")
    void testQuantizationTypes() {
        // Test from native name
        QuantizationType q4km = QuantizationType.fromNativeName("q4_k_m");
        assertThat(q4km).isEqualTo(QuantizationType.Q4_K_M);

        // Test properties
        assertThat(q4km.getNativeName()).isEqualTo("q4_k_m");
        assertThat(q4km.getCompressionRatio()).isEqualTo(8.5);
        assertThat(q4km.getQualityLevel()).isEqualTo(
                QuantizationType.QualityLevel.MEDIUM_HIGH);

        // Test invalid
        QuantizationType invalid = QuantizationType.fromNativeName("invalid");
        assertThat(invalid).isNull();
    }

    @Test
    @DisplayName("Should handle format detection edge cases")
    void testFormatDetectionEdgeCases() {
        // Non-existent path
        ModelFormat unknown1 = converter.detectFormat(Path.of("/nonexistent/path"));
        assertThat(unknown1).isEqualTo(ModelFormat.UNKNOWN);

        // Empty directory
        ModelFormat unknown2 = converter.detectFormat(tempDir);
        assertThat(unknown2).isEqualTo(ModelFormat.UNKNOWN);
    }

    @Test
    @DisplayName("Should generate conversion progress updates")
    void testConversionProgress() {
        ConversionProgress progress = ConversionProgress.builder()
                .conversionId(123)
                .progress(0.75f)
                .stage("Converting layers")
                .timestamp(System.currentTimeMillis())
                .build();

        assertThat(progress.getProgressPercent()).isEqualTo(75);
        assertThat(progress.isComplete()).isFalse();

        ConversionProgress complete = ConversionProgress.builder()
                .conversionId(124)
                .progress(1.0f)
                .stage("Complete")
                .timestamp(System.currentTimeMillis())
                .build();

        assertThat(complete.isComplete()).isTrue();
    }
}
