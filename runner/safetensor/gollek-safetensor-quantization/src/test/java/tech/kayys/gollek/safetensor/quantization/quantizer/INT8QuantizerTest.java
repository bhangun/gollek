/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * INT8QuantizerTest.java
 * ───────────────────────
 * Tests for INT8Quantizer.
 */
package tech.kayys.gollek.safetensor.quantization.quantizer;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for INT8Quantizer.
 */
class INT8QuantizerTest {

    private INT8Quantizer quantizer;

    @BeforeEach
    void setUp() {
        quantizer = new INT8Quantizer();
    }

    @Test
    void testGetName() {
        assertEquals("INT8Quantizer", quantizer.getName());
    }

    @Test
    void testSupportsInt8() {
        QuantConfig config = QuantConfig.int8();

        assertTrue(quantizer.supports(config));
    }

    @Test
    void testDoesNotSupportInt4() {
        QuantConfig config = QuantConfig.int4Gptq();

        assertFalse(quantizer.supports(config));
    }

    @Test
    void testDoesNotSupportFP8() {
        QuantConfig config = QuantConfig.fp8();

        assertFalse(quantizer.supports(config));
    }

    @Test
    void testNullTensorRejected() {
        QuantConfig config = QuantConfig.int8();

        assertThrows(IllegalArgumentException.class, () -> {
            quantizer.quantizeTensor(null, config);
        });
    }

    @Test
    void testPerChannelConfig() {
        QuantConfig config = QuantConfig.builder()
                .strategy(QuantizationEngine.QuantStrategy.INT8)
                .perChannel(true)
                .build();

        assertTrue(config.isPerChannel());
        assertTrue(quantizer.supports(config));
    }

    @Test
    void testPerTensorConfig() {
        QuantConfig config = QuantConfig.builder()
                .strategy(QuantizationEngine.QuantStrategy.INT8)
                .perChannel(false)
                .build();

        assertFalse(config.isPerChannel());
        assertTrue(quantizer.supports(config));
    }
}
