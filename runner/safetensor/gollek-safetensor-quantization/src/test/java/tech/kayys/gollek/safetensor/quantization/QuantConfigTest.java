/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantConfigTest.java
 * ───────────────────────
 * Tests for QuantConfig.
 */
package tech.kayys.gollek.safetensor.quantization;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine.QuantStrategy;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for QuantConfig.
 */
class QuantConfigTest {

    @Test
    void testDefaultConstructor() {
        QuantConfig config = new QuantConfig();

        assertEquals(QuantStrategy.INT4, config.getStrategy());
        assertEquals(128, config.getGroupSize());
        assertEquals(4, config.getBits());
        assertFalse(config.isSymmetric());
        assertTrue(config.isPerChannel());
        assertFalse(config.isActOrder());
        assertEquals(0.01, config.getDampPercent());
        assertEquals(128, config.getNumSamples());
        assertEquals(2048, config.getSeqLen());
        assertFalse(config.isDescAct());
    }

    @Test
    void testInt4GptqFactory() {
        QuantConfig config = QuantConfig.int4Gptq();

        assertEquals(QuantStrategy.INT4, config.getStrategy());
        assertEquals(4, config.getBits());
        assertEquals(128, config.getGroupSize());
    }

    @Test
    void testInt8Factory() {
        QuantConfig config = QuantConfig.int8();

        assertEquals(QuantStrategy.INT8, config.getStrategy());
        assertEquals(8, config.getBits());
        assertTrue(config.isPerChannel());
    }

    @Test
    void testFp8Factory() {
        QuantConfig config = QuantConfig.fp8();

        assertEquals(QuantStrategy.FP8, config.getStrategy());
        assertTrue(config.isSymmetric());
    }

    @Test
    void testBuilder() {
        QuantConfig config = QuantConfig.builder()
                .strategy(QuantStrategy.INT4)
                .groupSize(64)
                .bits(4)
                .symmetric(true)
                .perChannel(false)
                .actOrder(true)
                .dampPercent(0.05)
                .numSamples(256)
                .seqLen(4096)
                .descAct(true)
                .build();

        assertEquals(QuantStrategy.INT4, config.getStrategy());
        assertEquals(64, config.getGroupSize());
        assertEquals(4, config.getBits());
        assertTrue(config.isSymmetric());
        assertFalse(config.isPerChannel());
        assertTrue(config.isActOrder());
        assertEquals(0.05, config.getDampPercent());
        assertEquals(256, config.getNumSamples());
        assertEquals(4096, config.getSeqLen());
        assertTrue(config.isDescAct());
    }

    @Test
    void testBuilderWithDefaults() {
        QuantConfig config = QuantConfig.builder().build();

        assertEquals(QuantStrategy.INT4, config.getStrategy());
        assertEquals(128, config.getGroupSize());
        assertEquals(4, config.getBits());
    }

    @Test
    void testNullStrategyRejected() {
        assertThrows(NullPointerException.class, () -> {
            new QuantConfig(null, 128, 4, false, true, false, 0.01, 128, 2048, false);
        });
    }
}
