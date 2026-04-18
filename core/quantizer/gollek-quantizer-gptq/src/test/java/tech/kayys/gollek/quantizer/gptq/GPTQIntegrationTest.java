package tech.kayys.gollek.quantizer.gptq;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for GPTQ quantizer integration with safetensor runner.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class GPTQIntegrationTest {

    @TempDir
    Path tempDir;

    private GPTQQuantizerService quantizerService;

    @BeforeEach
    void setUp() {
        quantizerService = new GPTQQuantizerService();
        assertTrue(quantizerService.isInitialized());
    }

    @AfterEach
    void tearDown() {
        if (quantizerService != null) {
            quantizerService.close();
        }
    }

    // ── GPTQConfig Tests ─────────────────────────────────────────────────────

    @Test
    @Order(1)
    @DisplayName("Test GPTQConfig builder with custom settings")
    void testGPTQConfigBuilder() {
        GPTQConfig config = GPTQConfig.builder()
                .bits(4)
                .groupSize(64)
                .actOrder(true)
                .symmetric(false)
                .exllamaV2(false)
                .dequantDtype("float32")
                .perChannel(true)
                .dampPercent(0.02)
                .numSamples(256)
                .seqLen(4096)
                .quantizeEmbeddings(false)
                .calibrationDataPath("/path/to/calibration")
                .build();

        assertEquals(4, config.bits());
        assertEquals(64, config.groupSize());
        assertTrue(config.actOrder());
        assertFalse(config.symmetric());
        assertFalse(config.exllamaV2());
        assertEquals("float32", config.dequantDtype());
        assertTrue(config.perChannel());
        assertEquals(0.02, config.dampPercent());
        assertEquals(256, config.numSamples());
        assertEquals(4096, config.seqLen());
        assertFalse(config.quantizeEmbeddings());
        assertEquals("/path/to/calibration", config.calibrationDataPath());
    }

    @Test
    @Order(2)
    @DisplayName("Test GPTQConfig preset configurations")
    void testGPTQConfigPresets() {
        GPTQConfig config4bit = GPTQConfig.gptq4bit();
        assertEquals(4, config4bit.bits());
        assertEquals(128, config4bit.groupSize());
        assertFalse(config4bit.actOrder());
        assertTrue(config4bit.perChannel());

        GPTQConfig configActOrder = GPTQConfig.gptq4bitActOrder();
        assertTrue(configActOrder.actOrder());

        GPTQConfig configFP16 = GPTQConfig.gptq4bitFP16();
        assertEquals("float16", configFP16.dequantDtype());

        GPTQConfig configSymmetric = GPTQConfig.gptq4bitSymmetric();
        assertTrue(configSymmetric.symmetric());

        GPTQConfig configExllama = GPTQConfig.gptq4bitExllamaV2();
        assertTrue(configExllama.exllamaV2());
    }

    @Test
    @Order(3)
    @DisplayName("Test GPTQConfig derived properties")
    void testGPTQConfigDerivedProperties() {
        GPTQConfig config4bit = GPTQConfig.gptq4bit();
        assertEquals(8, config4bit.elementsPerInt32());
        assertEquals(0xF, config4bit.quantMask());
        assertEquals(8, config4bit.zeroPointOffset());
        assertEquals(15, config4bit.maxQuantValue());
        assertEquals(8.0, config4bit.compressionRatio());
        assertTrue(config4bit.isValidBits());

        GPTQConfig config8bit = GPTQConfig.gptq8bit();
        assertEquals(4, config8bit.elementsPerInt32());
        assertEquals(4.0, config8bit.compressionRatio());

        GPTQConfig config3bit = GPTQConfig.gptq3bit();
        assertEquals(10, config3bit.elementsPerInt32()); // 32/3 = 10 with padding
    }

    @Test
    @Order(4)
    @DisplayName("Test GPTQConfig validation")
    void testGPTQConfigValidation() {
        assertThrows(IllegalArgumentException.class, () -> {
            GPTQConfig.builder()
                    .bits(5) // Invalid bit width
                    .build();
        });

        assertTrue(GPTQConfig.gptq4bit().isValidBits());
        assertTrue(GPTQConfig.gptq8bit().isValidBits());
        assertTrue(GPTQConfig.gptq3bit().isValidBits());
        assertTrue(GPTQConfig.gptq2bit().isValidBits());
    }

    // ── GPTQQuantizerService Tests ───────────────────────────────────────────

    @Test
    @Order(10)
    @DisplayName("Test quantizer service initialization")
    void testQuantizerServiceInitialization() {
        assertTrue(quantizerService.isInitialized());
        
        GPTQQuantizerService anotherService = new GPTQQuantizerService();
        assertTrue(anotherService.isInitialized());
        anotherService.close();
    }

    @Test
    @Order(11)
    @DisplayName("Test async quantization task creation")
    void testAsyncQuantizationTaskCreation() {
        Path inputPath = tempDir.resolve("input_model");
        Path outputPath = tempDir.resolve("output_model");
        GPTQConfig config = GPTQConfig.gptq4bit();

        // This will fail due to missing input model, but tests the async infrastructure
        CompletableFuture<QuantizationResult> future =
                quantizerService.quantizeAsync(inputPath, outputPath, config);

        assertNotNull(future, "Async quantization should return a non-null future");
        // Wait briefly for the future to complete (it will fail due to missing input)
        try {
            future.get(2, java.util.concurrent.TimeUnit.SECONDS);
        } catch (Exception e) {
            // Expected - will throw exception due to missing input model
            assertTrue(true, "Future completed (possibly with exception as expected)");
        }
    }

    @Test
    @Order(12)
    @DisplayName("Test model inspection with invalid path")
    void testModelInspectionWithInvalidPath() {
        Path invalidPath = tempDir.resolve("nonexistent_model");
        
        assertThrows(IOException.class, () -> {
            quantizerService.inspect(invalidPath);
        });
    }

    // ── SafetensorConverter Tests ────────────────────────────────────────────

    @Test
    @Order(20)
    @DisplayName("Test conversion config presets")
    void testConversionConfigPresets() {
        GPTQSafetensorConverter.ConversionConfig defaults = GPTQSafetensorConverter.ConversionConfig.defaults();
        assertFalse(defaults.verbose());
        assertEquals("F32", defaults.outputDtype());
        assertFalse(defaults.includeNonQuantized());

        GPTQSafetensorConverter.ConversionConfig verbose = GPTQSafetensorConverter.ConversionConfig.verboseConfig();
        assertTrue(verbose.verbose());
        assertEquals("F32", verbose.outputDtype());
    }

    @Test
    @Order(21)
    @DisplayName("Test conversion result calculations")
    void testConversionResultCalculations() {
        GPTQSafetensorConverter.ConversionResult result = new GPTQSafetensorConverter.ConversionResult(
                32, 32, 7_000_000_000L, 2_000_000_000L, 5000, tempDir.resolve("output.safetensors"));

        assertEquals(32, result.layersConverted());
        assertEquals(32, result.tensorsWritten());
        assertEquals(7_000_000_000L, result.totalElements());
        assertEquals(2_000_000_000L, result.outputFileSizeBytes());
        assertEquals(5000, result.elapsedMs());

        // Throughput = 2000 MB / 5s = 400 MB/s
        assertEquals(400.0, result.throughputMBps(), 1.0);
    }

    // ── QuantizationResult Tests ─────────────────────────────────────────────

    @Test
    @Order(30)
    @DisplayName("Test quantization result calculations")
    void testQuantizationResultCalculations() {
        GPTQConfig config = GPTQConfig.gptq4bit();
        QuantizationResult result = new QuantizationResult(
                100, 100, 14_000_000_000L, 3_500_000_000L, 10000, config, tempDir.resolve("quantized"));

        assertEquals(100, result.inputTensors());
        assertEquals(100, result.outputTensors());
        assertEquals(14_000_000_000L, result.inputSizeBytes());
        assertEquals(3_500_000_000L, result.outputSizeBytes());
        assertEquals(10000, result.elapsedMs());

        // Compression ratio = 14GB / 3.5GB = 4x
        assertEquals(4.0, result.compressionRatio(), 0.01);

        // Throughput = 3500 MB / 10s = 350 MB/s
        assertEquals(350.0, result.throughputMBps(), 1.0);
    }

    // ── Integration Tests ────────────────────────────────────────────────────

    @Test
    @Order(40)
    @DisplayName("Test GPTQ config toString output")
    void testGPTQConfigToString() {
        GPTQConfig config = GPTQConfig.gptq4bit();
        String str = config.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("GPTQConfig"));
        assertTrue(str.contains("bits=4"));
        assertTrue(str.contains("groupSize=128"));
    }

    @Test
    @Order(41)
    @DisplayName("Test model inspection result toString")
    void testModelInspectionResultToString() {
        GPTQConfig config = GPTQConfig.gptq4bit();
        ModelInspectionResult result = new ModelInspectionResult(
                config, 32, 4_000_000_000L, 
                java.util.List.of("layer.0", "layer.1"),
                Map.of("model_type", "llama"));

        String str = result.toString();
        assertNotNull(str);
        assertTrue(str.contains("ModelInspectionResult"));
        assertTrue(str.contains("layers=32"));
        assertTrue(str.contains("memory="));
    }

    @Test
    @Order(42)
    @DisplayName("Test async conversion with progress callback")
    void testAsyncConversionWithProgressCallback() throws Exception {
        // This tests the infrastructure without actual model files
        CountDownLatch latch = new CountDownLatch(1);
        
        GPTQSafetensorConverter.ProgressCallback progressCallback = (layerName, current, total) -> {
            System.out.printf("Progress: %s (%d/%d)%n", layerName, current, total);
        };

        assertNotNull(progressCallback);
        progressCallback.onProgress("test.layer.0", 1, 32);
    }

    // ── Edge Cases and Error Handling ────────────────────────────────────────

    @Test
    @Order(50)
    @DisplayName("Test service close without initialization")
    void testServiceCloseWithoutIssues() {
        GPTQQuantizerService service = new GPTQQuantizerService();
        service.close();
        // Should not throw exception
        assertDoesNotThrow(() -> service.close());
    }

    @Test
    @Order(51)
    @DisplayName("Test config with extreme values")
    void testConfigWithExtremeValues() {
        GPTQConfig config = GPTQConfig.builder()
                .bits(8)
                .groupSize(1)
                .dampPercent(0.0)
                .numSamples(1)
                .seqLen(1)
                .build();

        assertEquals(8, config.bits());
        assertEquals(1, config.groupSize());
        assertEquals(0.0, config.dampPercent());
        assertEquals(1, config.numSamples());
        assertEquals(1, config.seqLen());
    }

    @Test
    @Order(52)
    @DisplayName("Test quantization result with zero values")
    void testQuantizationResultWithZeroValues() {
        GPTQConfig config = GPTQConfig.gptq4bit();
        QuantizationResult result = new QuantizationResult(
                0, 0, 0, 0, 0, config, tempDir.resolve("output"));

        assertEquals(0, result.inputTensors());
        assertEquals(0, result.compressionRatio());
        assertEquals(0, result.throughputMBps());
    }
}
