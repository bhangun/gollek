/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.ModelConfig;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardMetalHalfMatvecPairPolicyTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void halfMatvecPairExplicitEnableHonorsCoreMaxOutputAndDisableWins() throws Exception {
        Fixture generic = fixture("llama", false, false, false, 30, 4096, 4096);
        DirectForwardMetalHalfMatvecCoreOptions core =
                DirectForwardMetalHalfMatvecCoreOptions.defaults().withHalfMatvec(null, false, null, 4);

        assertTrue(policy(DirectForwardMetalHalfMatvecPairOptions.defaults()
                .withHalfMatvecPair(true, false), core)
                .shouldUseMetalHalfMatvecPair(generic.traits(), generic.config(), 4));
        assertFalse(policy(DirectForwardMetalHalfMatvecPairOptions.defaults()
                .withHalfMatvecPair(true, false), core)
                .shouldUseMetalHalfMatvecPair(generic.traits(), generic.config(), 5));
        assertFalse(policy(DirectForwardMetalHalfMatvecPairOptions.defaults()
                .withHalfMatvecPair(true, true), core)
                .shouldUseMetalHalfMatvecPair(generic.traits(), generic.config(), 4));
    }

    @Test
    void halfMatvecPairDefaultsToCoreAutoPolicy() throws Exception {
        Fixture generic = fixture("llama", false, false, false, 30, 4096, 4096);
        Fixture unknown = fixture("", false, false, false, 32, 4096, 11008);
        DirectForwardMetalHalfMatvecCoreOptions core =
                DirectForwardMetalHalfMatvecCoreOptions.defaults().withHalfMatvec(null, false, null, 4);

        assertTrue(policy(DirectForwardMetalHalfMatvecPairOptions.defaults(), core)
                .shouldUseMetalHalfMatvecPair(generic.traits(), generic.config(), 4));
        assertFalse(policy(DirectForwardMetalHalfMatvecPairOptions.defaults(), core)
                .shouldUseMetalHalfMatvecPair(generic.traits(), generic.config(), 5));
        assertFalse(policy(DirectForwardMetalHalfMatvecPairOptions.defaults(), core)
                .shouldUseMetalHalfMatvecPair(unknown.traits(), unknown.config(), 4));
    }

    @Test
    void halfLinearPairDefaultsToGenericAndNativeBf16ShapeRules() throws Exception {
        Fixture generic = fixture("llama", false, false, false, 30, 4096, 4096);
        Fixture gemma4 = fixture("gemma4_text", true, false, false, 40, 3584, 14336);

        assertTrue(defaultPolicy()
                .shouldUseMetalHalfLinearPair(generic.traits(), false, true));
        assertFalse(defaultPolicy()
                .shouldUseMetalHalfLinearPair(gemma4.traits(), false, true));
        assertTrue(defaultPolicy()
                .shouldUseMetalHalfLinearPair(gemma4.traits(), true, true));
        assertTrue(defaultPolicy()
                .shouldUseMetalHalfLinearPair(gemma4.traits(), false, false));
    }

    @Test
    void halfLinearPairExplicitAndDisableFlagsWin() throws Exception {
        Fixture generic = fixture("llama", false, false, false, 30, 4096, 4096);

        assertFalse(policy(DirectForwardMetalHalfMatvecPairOptions.defaults()
                .withHalfLinearPair(false, false))
                .shouldUseMetalHalfLinearPair(generic.traits(), true, false));
        assertFalse(policy(DirectForwardMetalHalfMatvecPairOptions.defaults()
                .withHalfLinearPair(true, true))
                .shouldUseMetalHalfLinearPair(generic.traits(), true, false));
    }

    private static Fixture fixture(String modelType, boolean gemma4Text, boolean gemma3Text, boolean qwenText,
            int layers, int hiddenSize, int intermediateSize) throws Exception {
        ModelConfig config = MAPPER.readValue("""
                {
                  "model_type": "%s",
                  "num_hidden_layers": %d,
                  "hidden_size": %d,
                  "intermediate_size": %d
                }
                """.formatted(modelType, layers, hiddenSize, intermediateSize), ModelConfig.class);
        ModelConfigTraits traits =
                new ModelConfigTraits(config, modelType, 0, 0, gemma4Text, gemma3Text, qwenText, false, false, false, false);
        return new Fixture(config, traits);
    }

    private record Fixture(ModelConfig config, ModelConfigTraits traits) {
    }

    private static DirectForwardMetalHalfMatvecPairPolicy defaultPolicy() {
        return policy(DirectForwardMetalHalfMatvecPairOptions.defaults());
    }

    private static DirectForwardMetalHalfMatvecPairPolicy policy(DirectForwardMetalHalfMatvecPairOptions options) {
        return policy(options, DirectForwardMetalHalfMatvecCoreOptions.defaults());
    }

    private static DirectForwardMetalHalfMatvecPairPolicy policy(
            DirectForwardMetalHalfMatvecPairOptions options,
            DirectForwardMetalHalfMatvecCoreOptions coreOptions) {
        return DirectForwardMetalHalfMatvecPairPolicy.from(options, coreOptions);
    }
}
