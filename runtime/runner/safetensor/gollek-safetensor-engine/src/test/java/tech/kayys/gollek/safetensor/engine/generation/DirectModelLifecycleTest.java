/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.jboss.logging.Logger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectModelLifecycleTest {

    @AfterEach
    void clearProfile() {
        DirectInferenceProfiler.clearProfile();
    }

    @Test
    void reusesLoadedModelWhenQuantizationStrategyMatches() {
        AtomicInteger loads = new AtomicInteger();
        AtomicInteger warmups = new AtomicInteger();
        DirectModelLifecycle<FakeLoadedModel> lifecycle = lifecycle(loads, warmups, new ArrayList<>());
        Path modelPath = Path.of("/tmp/gollek/model");

        String firstKey = lifecycle.load(modelPath, null);
        String secondKey = lifecycle.load(Path.of("/tmp/gollek/../gollek/model"),
                QuantizationEngine.QuantStrategy.NONE);

        assertEquals(firstKey, secondKey);
        assertEquals(1, loads.get());
        assertEquals(1, warmups.get());
        assertSame(lifecycle.find(modelPath), lifecycle.findByKey(firstKey));
        assertTrue(lifecycle.contains(modelPath));
    }

    @Test
    void reloadsAndReleasesExistingModelWhenQuantizationStrategyChanges() {
        AtomicInteger loads = new AtomicInteger();
        AtomicInteger warmups = new AtomicInteger();
        List<String> released = new ArrayList<>();
        DirectModelLifecycle<FakeLoadedModel> lifecycle = lifecycle(loads, warmups, released);
        Path modelPath = Path.of("/tmp/gollek/model");

        String baseKey = lifecycle.load(modelPath, QuantizationEngine.QuantStrategy.NONE);
        String quantizedKey = lifecycle.load(modelPath, QuantizationEngine.QuantStrategy.INT4);

        assertEquals("model", baseKey);
        assertEquals("model#int4", quantizedKey);
        assertEquals(2, loads.get());
        assertEquals(2, warmups.get());
        assertEquals(List.of(baseKey), released);
        assertNull(lifecycle.findByKey(baseKey));
        assertEquals(quantizedKey, lifecycle.find(modelPath).key());
    }

    @Test
    void unloadRemovesModelByPathAndKey() {
        AtomicInteger loads = new AtomicInteger();
        List<String> released = new ArrayList<>();
        DirectModelLifecycle<FakeLoadedModel> lifecycle = lifecycle(loads, new AtomicInteger(), released);
        Path modelPath = Path.of("/tmp/gollek/model");
        String key = lifecycle.load(modelPath, null);

        lifecycle.unload(modelPath);

        assertFalse(lifecycle.contains(modelPath));
        assertNull(lifecycle.findByKey(key));
        assertEquals(List.of(key), released);
    }

    @Test
    void requireLoadsMissingModel() {
        AtomicInteger loads = new AtomicInteger();
        DirectModelLifecycle<FakeLoadedModel> lifecycle = lifecycle(loads, new AtomicInteger(), new ArrayList<>());
        Path modelPath = Path.of("/tmp/gollek/model");

        FakeLoadedModel model = lifecycle.require(modelPath, false, null);

        assertEquals("model", model.key());
        assertEquals(1, loads.get());
    }

    private static DirectModelLifecycle<FakeLoadedModel> lifecycle(
            AtomicInteger loads,
            AtomicInteger warmups,
            List<String> released) {
        return new DirectModelLifecycle<>(
                (path, strategy) -> {
                    loads.incrementAndGet();
                    return new FakeLoadedModel(
                            path,
                            DirectModelLoader.modelKey(path, strategy),
                            DirectModelLoader.normalizeQuantStrategy(strategy));
                },
                FakeLoadedModel::quantStrategy,
                ignored -> warmups.incrementAndGet(),
                model -> released.add(model.key()),
                Logger.getLogger(DirectModelLifecycleTest.class));
    }

    private record FakeLoadedModel(
            Path path,
            String key,
            QuantizationEngine.QuantStrategy quantStrategy) implements SafetensorEngine.LoadedModel {
        @Override
        public Map<String, ?> weights() {
            return Map.of();
        }

        @Override
        public Tokenizer tokenizer() {
            return null;
        }

        @Override
        public boolean isQuantized() {
            return quantStrategy != QuantizationEngine.QuantStrategy.NONE;
        }

        @Override
        public ModelConfig config() {
            return new ModelConfig();
        }
    }
}
