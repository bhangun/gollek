/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ModelConfigTest.java
 * ─────────────────────
 * Unit tests for model configuration.
 */
package tech.kayys.gollek.safetensor.spi;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.ModelConfig;
import java.lang.reflect.Field;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for ModelConfig.
 */
class ModelConfigTest {

    private void setField(ModelConfig config, String fieldName, Object value) {
        try {
            Field field = ModelConfig.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            field.set(config, value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void testDefaultValues() {
        ModelConfig config = new ModelConfig();
        setField(config, "modelType", "llama");
        setField(config, "hiddenSize", 4096);
        setField(config, "numAttentionHeads", 32);
        setField(config, "intermediateSize", 11008);
        setField(config, "rmsNormEps", 1e-6);
        setField(config, "vocabSize", 32000);
        setField(config, "maxPositionEmbeddings", 2048);

        assertEquals(32, config.resolvedNumKvHeads()); // Defaults to numAttentionHeads
        assertEquals(10000.0, config.ropeTheta()); // LLaMA default
        assertFalse(config.hasSlidingWindow());
        assertFalse(config.isMoe());
    }

    @Test
    void testSlidingWindow() {
        ModelConfig config = new ModelConfig();
        setField(config, "modelType", "mistral");
        setField(config, "hiddenSize", 4096);
        setField(config, "numAttentionHeads", 32);
        setField(config, "numKeyValueHeads", 8);
        setField(config, "slidingWindow", 4096);

        assertTrue(config.hasSlidingWindow());
        assertEquals(4096, config.slidingWindowSize());
    }

    @Test
    void testMoEConfig() {
        ModelConfig config = new ModelConfig();
        setField(config, "modelType", "mixtral");
        setField(config, "numLocalExperts", 8);
        setField(config, "numExpertsPerTok", 2);
        setField(config, "decoderSparseStep", 1);

        assertTrue(config.isMoe());
        assertEquals(8, config.numLocalExperts());
        assertEquals(2, config.numExpertsPerTok());
        assertTrue(config.isMoeLayer(0));
        assertTrue(config.isMoeLayer(1));
    }

    @Test
    void testInterleavedMoE() {
        ModelConfig config = new ModelConfig();
        setField(config, "modelType", "deepseek");
        setField(config, "numLocalExperts", 8);
        setField(config, "numExpertsPerTok", 2);
        setField(config, "decoderSparseStep", 2);

        // With decoderSparseStep=2, only every other layer is MoE
        assertFalse(config.isMoeLayer(0)); // Layer 1 (0-indexed)
        assertTrue(config.isMoeLayer(1)); // Layer 2
        assertFalse(config.isMoeLayer(2)); // Layer 3
        assertTrue(config.isMoeLayer(3)); // Layer 4
    }
}
