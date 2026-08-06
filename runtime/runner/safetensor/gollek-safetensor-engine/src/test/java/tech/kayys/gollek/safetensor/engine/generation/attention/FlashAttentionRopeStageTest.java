/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import java.lang.reflect.Proxy;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionRopeStageTest {

    @Test
    void gemma4UsesSplitHalfRopeByDefault() {
        FlashAttentionRopeStage stage = stage(FlashAttentionRopeOptions.defaults());

        assertFalse(stage.useInterleavedRope(gemma4Policy()));
    }

    @Test
    void injectedLegacyOptionCanRestoreGemma4InterleavedRopeEscapeHatch() {
        FlashAttentionRopeStage stage = stage(new FlashAttentionRopeOptions(true, false));

        assertTrue(stage.useInterleavedRope(gemma4Policy()));
    }

    @Test
    void injectedExperimentalSplitHalfOptionOverridesLegacyInterleavedEscapeHatch() {
        FlashAttentionRopeStage stage = stage(new FlashAttentionRopeOptions(true, true));

        assertFalse(stage.useInterleavedRope(gemma4Policy()));
    }

    @Test
    void legacyGemma4OptionDoesNotAffectClassicInterleavedModels() {
        FlashAttentionRopeStage stage = stage(new FlashAttentionRopeOptions(true, false));

        assertTrue(stage.useInterleavedRope(classicInterleavedPolicy()));
    }

    private static FlashAttentionRopeStage stage(FlashAttentionRopeOptions options) {
        return new FlashAttentionRopeStage(new RopeFrequencyCache(), options);
    }

    private static FlashAttentionModelPolicy gemma4Policy() {
        ModelRuntimeTraits traits = ModelRuntimeTraits.builder()
                .nativeBf16Matvec()
                .attention(ModelAttentionTraitsPolicy.nativeBf16Matvec())
                .build();
        return FlashAttentionModelPolicy.resolve(interleavedArchitecture(traits), new ModelConfig(), traits);
    }

    private static FlashAttentionModelPolicy classicInterleavedPolicy() {
        ModelRuntimeTraits traits = ModelRuntimeTraits.EMPTY;
        return FlashAttentionModelPolicy.resolve(interleavedArchitecture(traits), new ModelConfig(), traits);
    }

    private static ModelArchitecture interleavedArchitecture(ModelRuntimeTraits traits) {
        return (ModelArchitecture) Proxy.newProxyInstance(
                FlashAttentionRopeStageTest.class.getClassLoader(),
                new Class<?>[] { ModelArchitecture.class },
                (proxy, method, args) -> switch (method.getName()) {
                    case "usesNeoxRope" -> false;
                    case "runtimeTraits" -> traits;
                    case "supportedArchClassNames", "supportedModelTypes" -> List.of();
                    case "toString" -> "interleavedArchitecture";
                    default -> defaultValue(method.getReturnType());
                });
    }

    private static Object defaultValue(Class<?> returnType) {
        if (returnType == boolean.class) {
            return false;
        }
        if (returnType == int.class) {
            return 0;
        }
        if (returnType == float.class) {
            return 0.0f;
        }
        if (returnType == double.class) {
            return 0.0;
        }
        return null;
    }
}
