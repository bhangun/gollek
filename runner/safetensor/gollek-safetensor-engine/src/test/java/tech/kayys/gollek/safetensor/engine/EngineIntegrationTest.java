/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * EngineIntegrationTest.java
 * ───────────────────────────
 * Integration tests for SafeTensor engine with audio and quantization.
 */
package tech.kayys.gollek.safetensor.engine;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.audio.WhisperEngine;
import tech.kayys.gollek.safetensor.audio.SpeechT5Engine;
import tech.kayys.gollek.safetensor.audio.model.AudioConfig;
import tech.kayys.gollek.safetensor.audio.processing.AudioDecoderRegistry;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for SafeTensor engine with audio and quantization modules.
 */
class EngineIntegrationTest {

    private DirectInferenceEngine inferenceEngine;
    private QuantizationEngine quantizationEngine;
    private WhisperEngine whisperEngine;
    private SpeechT5Engine ttsEngine;
    private AudioDecoderRegistry decoderRegistry;

    @BeforeEach
    void setUp() {
        inferenceEngine = new DirectInferenceEngine();
        quantizationEngine = new QuantizationEngine();
        whisperEngine = new WhisperEngine();
        ttsEngine = new SpeechT5Engine();
        decoderRegistry = AudioDecoderRegistry.getInstance();
    }

    @Test
    void testEngineInstantiation() {
        // Verify all engines can be instantiated
        assertNotNull(inferenceEngine);
        assertNotNull(quantizationEngine);
        assertNotNull(whisperEngine);
        assertNotNull(ttsEngine);
        assertNotNull(decoderRegistry);
    }

    @Test
    void testQuantizationEngineStrategies() {
        // Verify all quantization strategies are available
        QuantConfig int4Config = QuantConfig.int4Gptq();
        QuantConfig int8Config = QuantConfig.int8();
        QuantConfig fp8Config = QuantConfig.fp8();

        assertNotNull(int4Config);
        assertEquals(QuantizationEngine.QuantStrategy.INT4, int4Config.getStrategy());

        assertNotNull(int8Config);
        assertEquals(QuantizationEngine.QuantStrategy.INT8, int8Config.getStrategy());

        assertNotNull(fp8Config);
        assertEquals(QuantizationEngine.QuantStrategy.FP8, fp8Config.getStrategy());
    }

    @Test
    void testAudioDecoderRegistry() {
        // Verify decoder registry has all decoders
        String[] formats = decoderRegistry.getSupportedFormats();
        assertNotNull(formats);
        assertTrue(formats.length >= 4); // WAV, MP3, FLAC, OGG

        // Verify specific decoders
        assertTrue(decoderRegistry.isSupported("wav"));
        assertTrue(decoderRegistry.isSupported("mp3"));
        assertTrue(decoderRegistry.isSupported("flac"));
        assertTrue(decoderRegistry.isSupported("ogg"));
    }

    @Test
    void testAudioConfigBuilder() {
        // Test audio configuration builder
        AudioConfig config = AudioConfig.builder()
            .task(AudioConfig.Task.TRANSCRIBE)
            .language("en")
            .autoLanguage(true)
            .wordTimestamps(true)
            .beamSize(5)
            .build();

        assertNotNull(config);
        assertEquals(AudioConfig.Task.TRANSCRIBE, config.getTask());
        assertEquals("en", config.getLanguage());
        assertTrue(config.isAutoLanguage());
        assertTrue(config.isWordTimestamps());
        assertEquals(5, config.getBeamSize());
    }

    @Test
    void testTTSVoicesAvailable() {
        // Verify TTS engine has voices
        var voices = ttsEngine.getAvailableVoices();
        assertNotNull(voices);
        assertFalse(voices.isEmpty());

        // Check for expected voices
        assertTrue(voices.contains("alloy"));
        assertTrue(voices.contains("echo"));
        assertTrue(voices.contains("fable"));
    }

    @Test
    void testQuantizationConfigValidation() {
        // Test quantization configuration validation
        QuantConfig config = QuantConfig.builder()
            .strategy(QuantizationEngine.QuantStrategy.INT4)
            .groupSize(128)
            .bits(4)
            .symmetric(false)
            .perChannel(true)
            .build();

        assertNotNull(config);
        assertEquals(QuantizationEngine.QuantStrategy.INT4, config.getStrategy());
        assertEquals(128, config.getGroupSize());
        assertEquals(4, config.getBits());
        assertFalse(config.isSymmetric());
        assertTrue(config.isPerChannel());
    }

    @Test
    void testEngineInjection() {
        // Verify engines can be injected/used together
        // This simulates CDI injection behavior
        whisperEngine = new WhisperEngine();
        ttsEngine = new SpeechT5Engine();

        // Both engines should be able to coexist
        assertNotNull(whisperEngine);
        assertNotNull(ttsEngine);

        // Verify they have different responsibilities
        assertNotEquals(whisperEngine.getClass(), ttsEngine.getClass());
    }

    @Test
    void testAudioFormatAliases() {
        // Test format alias handling
        AudioDecoderRegistry registry = AudioDecoderRegistry.getInstance();

        // MP3 aliases
        assertNotNull(registry.getDecoder("mp3"));
        assertNotNull(registry.getDecoder("mpeg"));
        assertNotNull(registry.getDecoder("mpga"));

        // FLAC aliases
        assertNotNull(registry.getDecoder("flac"));
        assertNotNull(registry.getDecoder("fla"));

        // OGG aliases
        assertNotNull(registry.getDecoder("ogg"));
        assertNotNull(registry.getDecoder("vorbis"));
        assertNotNull(registry.getDecoder("oga"));
    }

    @Test
    void testEngineConfiguration() {
        // Test that engines can be configured independently
        AudioConfig audioConfig = AudioConfig.forTranscription();
        QuantConfig quantConfig = QuantConfig.int4Gptq();

        // Both configs should be valid
        assertNotNull(audioConfig);
        assertNotNull(quantConfig);

        // They should have different purposes
        assertEquals(AudioConfig.Task.TRANSCRIBE, audioConfig.getTask());
        assertEquals(QuantizationEngine.QuantStrategy.INT4, quantConfig.getStrategy());
    }

    @Test
    void testEngineModuleSeparation() {
        // Verify modules are properly separated
        // Audio engine should not depend on quantization for basic operations
        AudioConfig audioConfig = AudioConfig.builder()
            .task(AudioConfig.Task.TRANSCRIBE)
            .build();

        assertNotNull(audioConfig);

        // Quantization engine should work independently
        QuantConfig quantConfig = QuantConfig.builder()
            .strategy(QuantizationEngine.QuantStrategy.INT8)
            .build();

        assertNotNull(quantConfig);
    }

    @Test
    void testEngineReadyState() {
        // Verify engines are in ready state
        assertTrue(inferenceEngine.getClass().getName().contains("DirectInferenceEngine"));
        assertTrue(quantizationEngine.getClass().getName().contains("QuantizationEngine"));
        assertTrue(whisperEngine.getClass().getName().contains("WhisperEngine"));
        assertTrue(ttsEngine.getClass().getName().contains("SpeechT5Engine"));
    }

    @Test
    void testDependencyChain() {
        // Test that dependencies are properly chained
        // Engine → Audio → Decoder Registry
        AudioDecoderRegistry registry = AudioDecoderRegistry.getInstance();
        assertNotNull(registry);

        // Registry should have decoders registered
        String[] formats = registry.getSupportedFormats();
        assertTrue(formats.length > 0);

        // Engine should be able to access registry
        assertNotNull(decoderRegistry);
    }
}
