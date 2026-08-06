/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.nio.file.Path;
import java.util.Collection;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectLoadedModelRegistryTest {

    @Test
    void registersAndFindsByNormalizedPathAndKey() {
        DirectLoadedModelRegistry<FakeLoadedModel> registry = new DirectLoadedModelRegistry<>();
        Path path = Path.of("/tmp/gollek/../gollek/model");
        FakeLoadedModel model = new FakeLoadedModel(path, "model-key");

        registry.register(path, model);

        assertSame(model, registry.find(Path.of("/tmp/gollek/model")));
        assertSame(model, registry.findByKey("model-key"));
        assertTrue(registry.contains(Path.of("/tmp/gollek/model")));
    }

    @Test
    void removesPathAndKeyTogether() {
        DirectLoadedModelRegistry<FakeLoadedModel> registry = new DirectLoadedModelRegistry<>();
        Path path = Path.of("/tmp/gollek/model");
        FakeLoadedModel model = new FakeLoadedModel(path, "model-key");
        registry.register(path, model);

        assertSame(model, registry.remove(path));

        assertFalse(registry.contains(path));
        assertNull(registry.findByKey("model-key"));
    }

    @Test
    void usesStableLockForEquivalentPaths() {
        DirectLoadedModelRegistry<FakeLoadedModel> registry = new DirectLoadedModelRegistry<>();

        assertSame(
                registry.lockFor(Path.of("/tmp/gollek/../gollek/model")),
                registry.lockFor(Path.of("/tmp/gollek/model")));
    }

    @Test
    void snapshotIsDetachedAndUnmodifiable() {
        DirectLoadedModelRegistry<FakeLoadedModel> registry = new DirectLoadedModelRegistry<>();
        FakeLoadedModel model = new FakeLoadedModel(Path.of("/tmp/gollek/model"), "model-key");
        registry.register(model.path(), model);

        Collection<FakeLoadedModel> snapshot = registry.snapshot();
        registry.remove(model.path());

        assertEquals(1, snapshot.size());
        assertThrows(UnsupportedOperationException.class, () -> snapshot.clear());
    }

    private record FakeLoadedModel(Path path, String key) implements SafetensorEngine.LoadedModel {
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
            return false;
        }

        @Override
        public ModelConfig config() {
            return new ModelConfig();
        }
    }
}
