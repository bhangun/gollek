/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.mapper.GgufMetadataMapper;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelConfigTraitsTest {

    @Test
    void familyFlagsComeFromResolvedRuntimeTraitsNotModelTypeHeuristics() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {"model_type":"gemma4_text","architectures":["NativeBf16ForCausalLM"]}
                """, ModelConfig.class);
        ModelArchitecture architecture = architectureReturning(ModelRuntimeTraits.builder()
                .qwenText()
                .build());

        ModelConfigTraits traits = ModelConfigTraits.create(config, architecture);

        assertFalse(traits.nativeBf16Matvec());
        assertFalse(traits.gemma3Text());
        assertTrue(traits.qwenText());
    }

    @Test
    void perLayerInputMetadataRemainsStructuralFallback() {
        ModelConfig config = new GgufMetadataMapper().fromGgufMetadata(Map.of(
                "general.architecture", "custom",
                "custom.embedding_length_per_layer_input", 128));
        ModelArchitecture architecture = architectureReturning(ModelRuntimeTraits.builder().build());

        ModelConfigTraits traits = ModelConfigTraits.create(config, architecture);

        assertTrue(traits.perLayerInputPath());
    }

    @Test
    void vocabOnlyPerLayerMetadataDoesNotClaimNativeBf16StylePerLayerInputs() {
        ModelConfig config = new GgufMetadataMapper().fromGgufMetadata(Map.of(
                "general.architecture", "gemma4_unified",
                "gemma4_unified.vocab_size_per_layer_input", 262144));
        ModelArchitecture architecture = architectureReturning(ModelRuntimeTraits.builder()
                .nativeBf16Matvec()
                .build());

        ModelConfigTraits traits = ModelConfigTraits.create(config, architecture);

        assertTrue(traits.nativeBf16Matvec());
        assertEquals(0, traits.getHiddenSizePerLayerInput());
        assertEquals(262144, traits.getVocabSizePerLayerInput());
        assertFalse(traits.perLayerInputPath());
    }

    private static ModelArchitecture architectureReturning(ModelRuntimeTraits traits) {
        return architectureProxy((proxy, method, args) -> switch (method.getName()) {
            case "runtimeTraits" -> traits;
            case "id" -> "model-config-traits-test";
            case "supportedArchClassNames", "supportedModelTypes" -> List.of();
            case "toString" -> "model-config-traits-test";
            case "hashCode" -> System.identityHashCode(proxy);
            case "equals" -> proxy == args[0];
            default -> throw new UnsupportedOperationException(method.toString());
        });
    }

    private static ModelArchitecture architectureProxy(InvocationHandler handler) {
        return (ModelArchitecture) Proxy.newProxyInstance(
                ModelArchitecture.class.getClassLoader(),
                new Class<?>[] { ModelArchitecture.class },
                handler);
    }
}
