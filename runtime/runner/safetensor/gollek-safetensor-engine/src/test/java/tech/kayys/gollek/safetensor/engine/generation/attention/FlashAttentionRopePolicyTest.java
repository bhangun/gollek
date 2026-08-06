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

class FlashAttentionRopePolicyTest {
    private static final String LEGACY_INTERLEAVED_GEMMA4_ROPE_PROPERTY =
            "gollek.safetensor.legacy_interleaved_gemma4_rope";
    private static final String EXPERIMENTAL_GEMMA4_SPLIT_HALF_ROPE_PROPERTY =
            "gollek.safetensor.experimental_gemma4_split_half_rope";

    @Test
    void gemma4UsesSplitHalfRopeByDefault() {
        FlashAttentionRopePolicy policy = FlashAttentionRopePolicy.from(FlashAttentionRopeOptions.defaults());

        assertFalse(policy.useInterleavedRope(gemma4Policy()));
    }

    @Test
    void injectedLegacyOptionCanRestoreGemma4InterleavedRopeEscapeHatch() {
        FlashAttentionRopePolicy policy = FlashAttentionRopePolicy.from(new FlashAttentionRopeOptions(true, false));

        assertTrue(policy.useInterleavedRope(gemma4Policy()));
    }

    @Test
    void injectedExperimentalSplitHalfOptionOverridesLegacyInterleavedEscapeHatch() {
        FlashAttentionRopePolicy policy = FlashAttentionRopePolicy.from(new FlashAttentionRopeOptions(true, true));

        assertFalse(policy.useInterleavedRope(gemma4Policy()));
    }

    @Test
    void legacyGemma4OptionDoesNotAffectClassicInterleavedModels() {
        FlashAttentionRopePolicy policy = FlashAttentionRopePolicy.from(new FlashAttentionRopeOptions(true, false));

        assertTrue(policy.useInterleavedRope(classicInterleavedPolicy()));
    }

    @Test
    void nullOptionsUseDefaults() {
        FlashAttentionRopePolicy policy = FlashAttentionRopePolicy.from(null);

        assertFalse(policy.useInterleavedRope(gemma4Policy()));
    }

    @Test
    void systemPropertiesControlRawRopeRequests() {
        String previousLegacy = System.getProperty(LEGACY_INTERLEAVED_GEMMA4_ROPE_PROPERTY);
        String previousSplitHalf = System.getProperty(EXPERIMENTAL_GEMMA4_SPLIT_HALF_ROPE_PROPERTY);
        try {
            System.setProperty(LEGACY_INTERLEAVED_GEMMA4_ROPE_PROPERTY, "true");
            System.setProperty(EXPERIMENTAL_GEMMA4_SPLIT_HALF_ROPE_PROPERTY, "true");

            FlashAttentionRopeOptions options = FlashAttentionRopeOptions.fromSystemProperties();

            assertTrue(options.legacyInterleavedGemma4Rope());
            assertTrue(options.experimentalGemma4SplitHalfRope());
        } finally {
            restoreProperty(LEGACY_INTERLEAVED_GEMMA4_ROPE_PROPERTY, previousLegacy);
            restoreProperty(EXPERIMENTAL_GEMMA4_SPLIT_HALF_ROPE_PROPERTY, previousSplitHalf);
        }
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
                FlashAttentionRopePolicyTest.class.getClassLoader(),
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

    private static void restoreProperty(String name, String previous) {
        if (previous == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, previous);
        }
    }
}
