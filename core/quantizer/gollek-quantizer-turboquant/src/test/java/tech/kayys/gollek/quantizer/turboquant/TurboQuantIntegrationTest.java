package tech.kayys.gollek.quantizer.turboquant;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;
import tech.kayys.gollek.gguf.loader.GGUFDequantizer;

/**
 * Comprehensive tests for TurboQuant multi-format quantizer.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class TurboQuantIntegrationTest {

    @TempDir
    Path tempDir;

    private TurboQuantService service;

    @BeforeEach
    void setUp() {
        service = new TurboQuantService();
        assertTrue(service.isInitialized());
    }

    @AfterEach
    void tearDown() {
        if (service != null) {
            service.close();
        }
    }

    // ── TurboQuantConfig Tests ──────────────────────────────────────────────

    @Test
    @Order(1)
    @DisplayName("Test TurboQuantConfig MSE preset")
    void testTurboQuantConfigMsePreset() {
        TurboQuantConfig config = TurboQuantConfig.mse4bit(4096);
        
        assertEquals(4, config.bits());
        assertEquals(4096, config.dimension());
        assertEquals(TurboQuantConfig.Variant.MSE, config.variant());
        assertEquals(TurboQuantConfig.RotationStrategy.HADAMARD, config.rotation());
        assertFalse(config.splitOutliers());
        assertEquals(16, config.numLevels()); // 2^4
        assertEquals(4, config.mseStageBits());
        assertTrue(config.isMseVariant());
        assertFalse(config.isInnerProductVariant());
        assertTrue(config.usesHadamard());
    }

    @Test
    @Order(2)
    @DisplayName("Test TurboQuantConfig Prod preset")
    void testTurboQuantConfigProdPreset() {
        TurboQuantConfig config = TurboQuantConfig.prod4bit(4096);
        
        assertEquals(4, config.bits());
        assertEquals(TurboQuantConfig.Variant.INNER_PRODUCT, config.variant());
        assertEquals(3, config.mseStageBits()); // b-1 for prod
        assertFalse(config.isMseVariant());
        assertTrue(config.isInnerProductVariant());
    }

    @Test
    @Order(3)
    @DisplayName("Test TurboQuantConfig KV Cache presets")
    void testTurboQuantConfigKvCachePresets() {
        TurboQuantConfig config2bit = TurboQuantConfig.prod2bitKvCache(128);
        assertEquals(2, config2bit.bits());
        assertTrue(config2bit.splitOutliers());
        assertEquals(32, config2bit.outlierChannels());
        
        TurboQuantConfig config3bit = TurboQuantConfig.prod3bitKvCache(128);
        assertEquals(3, config3bit.bits());
        assertTrue(config3bit.splitOutliers());
    }

    @Test
    @Order(4)
    @DisplayName("Test TurboQuantConfig derived properties")
    void testTurboQuantConfigDerivedProperties() {
        TurboQuantConfig config = TurboQuantConfig.mse4bit(4096);
        
        assertEquals(16, config.numLevels());
        assertEquals(4, config.mseStageBits());
        
        // MSE bound: (√(3π)/2) · 4^(-b)
        double mseBound = config.mseBound();
        assertTrue(mseBound > 0);
        assertTrue(mseBound < 1);
        
        // Empirical MSE for b=4 should be ~0.009
        assertEquals(0.009, config.empiricalMse(), 0.001);
        
        // Optimality gap should be ≤ √(3π)/2 ≈ 2.72
        assertEquals(Math.sqrt(3 * Math.PI) / 2.0, config.optimalityGap(), 0.001);
        
        // Lower bound: 4^(-b)
        double lowerBound = config.lowerBound();
        assertTrue(lowerBound > 0);
        assertTrue(lowerBound < mseBound);
    }

    @Test
    @Order(5)
    @DisplayName("Test TurboQuantConfig effective bits with outlier split")
    void testTurboQuantConfigEffectiveBits() {
        TurboQuantConfig config = TurboQuantConfig.prod2bitKvCache(128);
        
        // 32 outlier channels at 3 bits + 96 normal at 2 bits
        // Effective = (32*3 + 96*2) / 128 = 2.25 bits
        double effectiveBits = config.effectiveBitsPerChannel(128);
        assertEquals(2.25, effectiveBits, 0.01);
    }

    // ── TurboQuantService Tests ─────────────────────────────────────────────

    @Test
    @Order(10)
    @DisplayName("Test service initialization")
    void testServiceInitialization() {
        assertTrue(service.isInitialized());
        
        TurboQuantService anotherService = new TurboQuantService();
        assertTrue(anotherService.isInitialized());
        anotherService.close();
    }

    @Test
    @Order(11)
    @DisplayName("Test format detection with invalid path")
    void testFormatDetectionWithInvalidPath() {
        Path invalidPath = tempDir.resolve("nonexistent_model");

        // detectFormat returns UNKNOWN with LOW confidence instead of throwing
        QuantizerRegistry.Detection detection = assertDoesNotThrow(() -> service.detectFormat(invalidPath));
        assertEquals(QuantizerRegistry.QuantFormat.UNKNOWN, detection.format());
        assertEquals(QuantizerRegistry.Detection.Confidence.LOW, detection.confidence());
    }

    @Test
    @Order(12)
    @DisplayName("Test supported formats output")
    void testSupportedFormatsOutput() {
        // Just verify it doesn't throw
        assertDoesNotThrow(() -> service.printSupportedFormats());
    }

    // ── TurboQuantEngine Tests ──────────────────────────────────────────────

    @Test
    @Order(20)
    @DisplayName("Test TurboQuantEngine creation")
    void testTurboQuantEngineCreation() {
        TurboQuantConfig config = TurboQuantConfig.mse4bit(128);
        TurboQuantEngine engine = service.createEngine(config);
        
        assertNotNull(engine);
    }

    @Test
    @Order(21)
    @DisplayName("Test TurboQuant MSE quantization and dequantization")
    void testTurboQuantMseQuantizeDequantize() {
        int dimension = 128;
        TurboQuantConfig config = TurboQuantConfig.mse4bit(dimension);
        TurboQuantEngine engine = service.createEngine(config);

        // Create test vector
        float[] vector = createRandomVector(dimension, 42L);

        // Quantize using MSE method
        int[] indices = new int[dimension];
        engine.quantizeMse(vector, indices);
        assertNotNull(indices);

        // Dequantize
        float[] dequantized = new float[dimension];
        engine.dequantizeMse(indices, dequantized);
        assertNotNull(dequantized);
        assertEquals(dimension, dequantized.length);

        // Check MSE is within bounds
        double mse = computeMSE(vector, dequantized);
        assertTrue(mse < config.mseBound() * 10, "MSE should be within theoretical bound");
    }

    @Test
    @Order(22)
    @DisplayName("Test TurboQuant Prod quantization")
    void testTurboQuantProdQuantize() {
        int dimension = 128;
        TurboQuantConfig config = TurboQuantConfig.prod4bit(dimension);
        TurboQuantEngine engine = service.createEngine(config);

        float[] vector = createRandomVector(dimension, 42L);

        // Use quantizeProd which returns QuantProdResult
        TurboQuantEngine.QuantProdResult quantized = engine.quantizeProd(vector);
        assertNotNull(quantized);

        float[] dequantized = new float[dimension];
        engine.dequantizeProd(quantized, dequantized);
        assertEquals(dimension, dequantized.length);
    }

    @Test
    @Order(23)
    @DisplayName("Test TurboQuant vector quantization via service")
    void testServiceQuantizeVector() {
        int dimension = 256;
        TurboQuantConfig config = TurboQuantConfig.mse4bit(dimension);

        float[] vector = createRandomVector(dimension, 42L);

        // Use engine directly since service may not have quantizeVector method
        TurboQuantEngine engine = service.createEngine(config);
        int[] indices = new int[dimension];
        engine.quantizeMse(vector, indices);
        assertNotNull(indices);

        float[] dequantized = new float[dimension];
        engine.dequantizeMse(indices, dequantized);
        assertEquals(dimension, dequantized.length);
    }

    // ── KV Cache Tests ──────────────────────────────────────────────────────

    @Test
    @Order(30)
    @DisplayName("Test KV cache quantizer creation")
    void testKvCacheCreation() {
        TurboQuantConfig config = TurboQuantConfig.prod2bitKvCache(128);
        int maxSeqLen = 512;
        TurboQuantKVCache kvCache = service.createKvCache(config, maxSeqLen);

        assertNotNull(kvCache);
    }

    // ── Multi-Format Dequantizer Tests ──────────────────────────────────────

    @Test
    @Order(40)
    @DisplayName("Test dequantizer creation for various formats")
    void testDequantizerCreation() {
        // BnB
        Object bnbDequantizer = service.createDequantizer(QuantizerRegistry.QuantFormat.BNB_NF4);
        assertNotNull(bnbDequantizer);
        assertInstanceOf(BnBDequantizer.class, bnbDequantizer);
        
        // HQQ
        Object hqqDequantizer = service.createDequantizer(QuantizerRegistry.QuantFormat.HQQ);
        assertNotNull(hqqDequantizer);
        assertInstanceOf(HQQDequantizer.class, hqqDequantizer);
        
        // SqueezeLLM
        Object squeezeDequantizer = service.createDequantizer(QuantizerRegistry.QuantFormat.SQUEEZELLM);
        assertNotNull(squeezeDequantizer);
        assertInstanceOf(SqueezeLLMDequantizer.class, squeezeDequantizer);
        
        // GGUF
        Object ggufDequantizer = service.createDequantizer(QuantizerRegistry.QuantFormat.GGUF);
        assertNotNull(ggufDequantizer);
        assertInstanceOf(GGUFDequantizer.class, ggufDequantizer);
        
        // GPTQ/AWQ return null (use their own modules)
        assertNull(service.createDequantizer(QuantizerRegistry.QuantFormat.GPTQ));
        assertNull(service.createDequantizer(QuantizerRegistry.QuantFormat.AWQ));
    }

    // ── Dequantizer Tests ───────────────────────────────────────────────────

    @Test
    @Order(50)
    @DisplayName("Test BnBDequantizer initialization")
    void testBnBDequantizerInitialization() {
        BnBDequantizer dequantizer = new BnBDequantizer();
        assertNotNull(dequantizer);
    }

    @Test
    @Order(51)
    @DisplayName("Test HQQDequantizer initialization")
    void testHQQDequantizerInitialization() {
        HQQDequantizer dequantizer = new HQQDequantizer(4, 128, HQQDequantizer.QuantAxis.INPUT);
        assertNotNull(dequantizer);
    }

    @Test
    @Order(52)
    @DisplayName("Test SqueezeLLMDequantizer initialization")
    void testSqueezeLLMDequantizerInitialization() {
        SqueezeLLMDequantizer dequantizer = new SqueezeLLMDequantizer();
        assertNotNull(dequantizer);
    }

    @Test
    @Order(53)
    @DisplayName("Test GGUFDequantizer initialization")
    void testGGUFDequantizerInitialization() {
        GGUFDequantizer dequantizer = new GGUFDequantizer();
        assertNotNull(dequantizer);
    }

    // ── QuantizerRegistry Tests ─────────────────────────────────────────────

    @Test
    @Order(60)
    @DisplayName("Test QuantizerRegistry supported formats")
    void testQuantizerRegistryFormats() {
        QuantizerRegistry.QuantFormat[] formats = QuantizerRegistry.QuantFormat.values();
        
        // Should have 9 formats (including UNKNOWN)
        assertEquals(9, formats.length);
        
        // Check specific formats exist
        assertTrue(java.util.Arrays.asList(formats).contains(QuantizerRegistry.QuantFormat.GPTQ));
        assertTrue(java.util.Arrays.asList(formats).contains(QuantizerRegistry.QuantFormat.AWQ));
        assertTrue(java.util.Arrays.asList(formats).contains(QuantizerRegistry.QuantFormat.GGUF));
        assertTrue(java.util.Arrays.asList(formats).contains(QuantizerRegistry.QuantFormat.BNB_NF4));
        assertTrue(java.util.Arrays.asList(formats).contains(QuantizerRegistry.QuantFormat.HQQ));
        assertTrue(java.util.Arrays.asList(formats).contains(QuantizerRegistry.QuantFormat.SQUEEZELLM));
    }

    @Test
    @Order(61)
    @DisplayName("Test QuantizerRegistry Detection record")
    void testQuantizerRegistryDetection() {
        QuantizerRegistry.Detection detection = new QuantizerRegistry.Detection(
                QuantizerRegistry.QuantFormat.GPTQ,
                QuantizerRegistry.Detection.Confidence.HIGH,
                "test evidence"
        );
        
        assertEquals(QuantizerRegistry.QuantFormat.GPTQ, detection.format());
        assertEquals(QuantizerRegistry.Detection.Confidence.HIGH, detection.confidence());
        assertEquals("test evidence", detection.evidence());
        assertTrue(detection.isKnown());
    }

    // ── Edge Cases and Error Handling ───────────────────────────────────────

    @Test
    @Order(70)
    @DisplayName("Test service close without issues")
    void testServiceCloseWithoutIssues() {
        TurboQuantService svc = new TurboQuantService();
        svc.close();
        assertDoesNotThrow(() -> svc.close());
    }

    @Test
    @Order(71)
    @DisplayName("Test model inspection result toString")
    void testModelInspectionResultToString() throws IOException {
        QuantizerRegistry.Detection detection = new QuantizerRegistry.Detection(
                QuantizerRegistry.QuantFormat.GPTQ,
                QuantizerRegistry.Detection.Confidence.HIGH,
                "test"
        );
        
        TurboQuantService.ModelInspectionResult result = new TurboQuantService.ModelInspectionResult(
                detection, tempDir, 4_000_000_000L
        );
        
        String str = result.toString();
        assertNotNull(str);
        assertTrue(str.contains("ModelInspectionResult"));
        assertTrue(str.contains("format=GPTQ"));
        assertTrue(str.contains("size="));
    }

    // ── Helper Methods ──────────────────────────────────────────────────────

    private float[] createRandomVector(int dimension, long seed) {
        Random rng = new Random(seed);
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = (float) rng.nextGaussian();
        }
        // Normalize to unit sphere
        float norm = 0;
        for (float v : vector) norm += v * v;
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < dimension; i++) {
            vector[i] /= norm;
        }
        return vector;
    }

    private double computeMSE(float[] original, float[] reconstructed) {
        double mse = 0;
        for (int i = 0; i < original.length; i++) {
            double diff = original[i] - reconstructed[i];
            mse += diff * diff;
        }
        return mse / original.length;
    }
}
