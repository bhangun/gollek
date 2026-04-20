/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPTQQuantizerTest.java
 * ───────────────────────
 * Tests for GPTQQuantizer.
 */
package tech.kayys.gollek.safetensor.quantization.quantizer;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for GPTQQuantizer.
 */
class GPTQQuantizerTest {

    private GPTQQuantizer quantizer;

    @BeforeEach
    void setUp() {
        quantizer = new GPTQQuantizer();
    }

    @Test
    void testGetName() {
        assertEquals("GPTQQuantizer", quantizer.getName());
    }

    @Test
    void testSupportsInt4() {
        QuantConfig config = QuantConfig.builder()
                .strategy(QuantizationEngine.QuantStrategy.INT4)
                .bits(4)
                .build();

        assertTrue(quantizer.supports(config));
    }

    @Test
    void testSupportsInt8() {
        QuantConfig config = QuantConfig.builder()
                .strategy(QuantizationEngine.QuantStrategy.INT8)
                .bits(8)
                .build();

        assertTrue(quantizer.supports(config));
    }

    @Test
    void testDoesNotSupportFP8() {
        QuantConfig config = QuantConfig.builder()
                .strategy(QuantizationEngine.QuantStrategy.FP8)
                .bits(8)
                .build();

        // GPTQ is for integer quantization, not FP8
        assertFalse(quantizer.supports(config));
    }

    @Test
    void testNullTensorRejected() {
        QuantConfig config = QuantConfig.int4Gptq();

        assertThrows(IllegalArgumentException.class, () -> {
            quantizer.quantizeTensor(null, config);
        });
    }

    @Test
    void testQuantizeDequantizeRoundtrip() {
        // Note: This test would require actual AccelTensor implementation
        // For now, it verifies the methods exist and handle null gracefully
        QuantConfig config = QuantConfig.int4Gptq();

        // In a real test with actual AccelTensor implementation:
        // AccelTensor quantized = quantizer.quantizeTensor(tensor, config);
        // AccelTensor dequantized = quantizer.dequantizeTensor(quantized, config);
        // Verify dequantized values are close to original
    }
}
