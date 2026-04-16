/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AudioDecoderRegistryTest.java
 * ───────────────────────
 * Tests for audio decoder registry.
 */
package tech.kayys.gollek.safetensor.audio.processing;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for AudioDecoderRegistry.
 */
class AudioDecoderRegistryTest {

    private AudioDecoderRegistry registry;

    @BeforeEach
    void setUp() {
        registry = AudioDecoderRegistry.getInstance();
    }

    @Test
    void testSingletonInstance() {
        AudioDecoderRegistry instance1 = AudioDecoderRegistry.getInstance();
        AudioDecoderRegistry instance2 = AudioDecoderRegistry.getInstance();
        assertSame(instance1, instance2);
    }

    @Test
    void testGetWavDecoder() {
        AudioDecoder decoder = registry.getDecoder("wav");
        assertNotNull(decoder);
        assertEquals("wav", decoder.getFormat());
    }

    @Test
    void testGetMp3Decoder() {
        AudioDecoder decoder = registry.getDecoder("mp3");
        assertNotNull(decoder);
        assertEquals("mp3", decoder.getFormat());
    }

    @Test
    void testGetFlacDecoder() {
        AudioDecoder decoder = registry.getDecoder("flac");
        assertNotNull(decoder);
        assertEquals("flac", decoder.getFormat());
    }

    @Test
    void testGetOggDecoder() {
        AudioDecoder decoder = registry.getDecoder("ogg");
        assertNotNull(decoder);
        assertEquals("ogg", decoder.getFormat());
    }

    @Test
    void testFormatAliases() {
        // MP3 aliases
        assertNotNull(registry.getDecoder("mpeg"));
        assertNotNull(registry.getDecoder("mpga"));

        // FLAC aliases
        assertNotNull(registry.getDecoder("fla"));

        // OGG aliases
        assertNotNull(registry.getDecoder("vorbis"));
        assertNotNull(registry.getDecoder("oga"));

        // WAV aliases
        assertNotNull(registry.getDecoder("wave"));
        assertNotNull(registry.getDecoder("rifx"));
    }

    @Test
    void testIsSupported() {
        assertTrue(registry.isSupported("wav"));
        assertTrue(registry.isSupported("mp3"));
        assertTrue(registry.isSupported("flac"));
        assertTrue(registry.isSupported("ogg"));

        assertFalse(registry.isSupported("unknown_format"));
    }

    @Test
    void testGetSupportedFormats() {
        String[] formats = registry.getSupportedFormats();
        assertNotNull(formats);
        assertTrue(formats.length >= 4); // At least WAV, MP3, FLAC, OGG

        // Check for expected formats
        boolean hasWav = false, hasMp3 = false, hasFlac = false, hasOgg = false;
        for (String format : formats) {
            if ("wav".equals(format)) hasWav = true;
            if ("mp3".equals(format)) hasMp3 = true;
            if ("flac".equals(format)) hasFlac = true;
            if ("ogg".equals(format)) hasOgg = true;
        }

        assertTrue(hasWav, "Should support WAV");
        assertTrue(hasMp3, "Should support MP3");
        assertTrue(hasFlac, "Should support FLAC");
        assertTrue(hasOgg, "Should support OGG");
    }

    @Test
    void testNullFormat() {
        // Should default to WAV decoder
        AudioDecoder decoder = registry.getDecoder(null);
        assertNotNull(decoder);
        assertEquals("wav", decoder.getFormat());
    }

    @Test
    void testCaseInsensitive() {
        assertNotNull(registry.getDecoder("WAV"));
        assertNotNull(registry.getDecoder("Mp3"));
        assertNotNull(registry.getDecoder("FLAC"));
        assertNotNull(registry.getDecoder("OgG"));
    }
}
