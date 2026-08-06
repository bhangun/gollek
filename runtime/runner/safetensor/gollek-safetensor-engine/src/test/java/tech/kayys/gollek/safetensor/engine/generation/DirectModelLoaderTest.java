/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DirectModelLoaderTest {

    @Test
    void normalizesNullQuantizationStrategyToNone() {
        assertEquals(QuantizationEngine.QuantStrategy.NONE, DirectModelLoader.normalizeQuantStrategy(null));
        assertEquals(QuantizationEngine.QuantStrategy.INT4,
                DirectModelLoader.normalizeQuantStrategy(QuantizationEngine.QuantStrategy.INT4));
    }

    @Test
    void buildsStableUnquantizedModelKey() {
        assertEquals("Phi-4-mini-instruct",
                DirectModelLoader.modelKey(Path.of("/models/microsoft/Phi-4-mini-instruct"),
                        QuantizationEngine.QuantStrategy.NONE));
        assertEquals("Phi-4-mini-instruct",
                DirectModelLoader.modelKey(Path.of("/models/microsoft/Phi-4-mini-instruct"), null));
    }

    @Test
    void buildsQuantizedModelKeyWithStrategySuffix() {
        assertEquals("Phi-4-mini-instruct#int4",
                DirectModelLoader.modelKey(Path.of("/models/microsoft/Phi-4-mini-instruct"),
                        QuantizationEngine.QuantStrategy.INT4));
    }
}
