/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * FP8QuantizerTest.java
 * ───────────────────────
 * Tests for FP8Quantizer.
 */
package tech.kayys.gollek.safetensor.quantization.quantizer;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for FP8Quantizer.
 */
class FP8QuantizerTest {

    private FP8Quantizer quantizer;

    @BeforeEach
    void setUp() {
        quantizer = new FP8Quantizer();
    }

    @Test
    void testGetName() {
        assertEquals("FP8Quantizer", quantizer.getName());
    }

    @Test
    void testSupportsFP8() {
        QuantConfig config = QuantConfig.fp8();

        assertTrue(quantizer.supports(config));
    }

    @Test
    void testDoesNotSupportInt4() {
        QuantConfig config = QuantConfig.int4Gptq();

        assertFalse(quantizer.supports(config));
    }

    @Test
    void testDoesNotSupportInt8() {
        QuantConfig config = QuantConfig.int8();

        assertFalse(quantizer.supports(config));
    }

    @Test
    void testNullTensorRejected() {
        QuantConfig config = QuantConfig.fp8();

        assertThrows(IllegalArgumentException.class, () -> {
            quantizer.quantizeTensor(null, config);
        });
    }

    @Test
    void testE4M3Format() {
        QuantConfig config = QuantConfig.builder()
                .strategy(QuantizationEngine.QuantStrategy.FP8)
                .symmetric(true)
                .build();

        assertTrue(config.isSymmetric());
        assertTrue(quantizer.supports(config));
    }

    @Test
    void testE5M2Format() {
        QuantConfig config = QuantConfig.builder()
                .strategy(QuantizationEngine.QuantStrategy.FP8)
                .symmetric(false)
                .build();

        assertFalse(config.isSymmetric());
        assertTrue(quantizer.supports(config));
    }

    @Test
    void testFP8FormatEnum() {
        assertEquals(4, FP8Quantizer.FP8Format.E4M3.getExpBits());
        assertEquals(3, FP8Quantizer.FP8Format.E4M3.getMantissaBits());
        assertEquals(5, FP8Quantizer.FP8Format.E5M2.getExpBits());
        assertEquals(2, FP8Quantizer.FP8Format.E5M2.getMantissaBits());
    }
}
