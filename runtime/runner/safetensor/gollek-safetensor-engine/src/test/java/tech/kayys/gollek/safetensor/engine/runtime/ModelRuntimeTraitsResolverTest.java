/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.runtime;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelRuntimeTraitsResolverTest {

    @Test
    void providedTraitsWinOverArchitectureAndFallback() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {"model_type":"gemma4_text","architectures":["Gemma4ForCausalLM"]}
                """, ModelConfig.class);
        ModelRuntimeTraits provided = ModelRuntimeTraits.builder()
                .qwenText()
                .build();

        ModelRuntimeTraits traits = ModelRuntimeTraitsResolver.resolve(architectureThatMustNotBeUsed(), config,
                provided);

        assertTrue(traits.qwenText());
        assertFalse(traits.nativeBf16Matvec());
        assertEquals(ModelRuntimeTraits.QWEN_DEFAULT_SYSTEM_PROMPT, traits.defaultSystemPrompt());
    }

    @Test
    void architectureTraitsWinOverConfigFallbackAndPreserveDetectedModalities() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {"model_type":"qwen2_vl","architectures":["Qwen2VLForConditionalGeneration"]}
                """, ModelConfig.class);
        ModelRuntimeTraits architectureTraits = ModelRuntimeTraits.builder()
                .nativeBf16Matvec()
                .build();

        ModelRuntimeTraits traits = ModelRuntimeTraitsResolver.resolve(architectureReturning(architectureTraits),
                config);

        assertTrue(traits.nativeBf16Matvec());
        assertFalse(traits.qwenText());
        assertTrue(traits.visionModel());
        assertTrue(traits.multimodalModel());
    }

    @Test
    void fallsBackToConfigWhenArchitectureHasNoRuntimeTraits() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {"model_type":"gemma4_text","architectures":["Gemma4ForCausalLM"]}
                """, ModelConfig.class);

        ModelRuntimeTraits traits = ModelRuntimeTraitsResolver.resolve(architectureReturning(null), config);

        assertTrue(traits.nativeBf16Matvec());
        assertEquals(ModelRuntimeTraits.PromptBosPolicy.NEVER, traits.promptBosPolicy());
    }

    private static ModelArchitecture architectureReturning(ModelRuntimeTraits traits) {
        return architectureProxy((proxy, method, args) -> switch (method.getName()) {
            case "runtimeTraits" -> traits;
            case "id" -> "resolver-test";
            case "supportedArchClassNames", "supportedModelTypes" -> List.of();
            case "toString" -> "resolver-test";
            case "hashCode" -> System.identityHashCode(proxy);
            case "equals" -> proxy == args[0];
            default -> throw new UnsupportedOperationException(method.toString());
        });
    }

    private static ModelArchitecture architectureThatMustNotBeUsed() {
        return architectureProxy((proxy, method, args) -> {
            if ("runtimeTraits".equals(method.getName())) {
                throw new AssertionError("provided runtime traits should avoid architecture lookup");
            }
            throw new UnsupportedOperationException(method.toString());
        });
    }

    private static ModelArchitecture architectureProxy(InvocationHandler handler) {
        return (ModelArchitecture) Proxy.newProxyInstance(
                ModelArchitecture.class.getClassLoader(),
                new Class<?>[] { ModelArchitecture.class },
                handler);
    }
}
