/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardMetalHalfMatvecAutoPolicyTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void defaultAutoAllowsGenericLargeShapeWithinThreshold() throws Exception {
        DirectForwardMetalHalfMatvecAutoPolicy policy = policy(null, 4);
        Fixture generic = fixture("llama", false, false, false, false, 30, 4096, 4096);

        assertTrue(policy.shouldUseMetalHalfMatvec(generic.traits(), generic.config(), 4));
        assertFalse(policy.shouldUseMetalHalfMatvec(generic.traits(), generic.config(), 5));
    }

    @Test
    void explicitAutoDisableWinsForCandidateModel() throws Exception {
        DirectForwardMetalHalfMatvecAutoPolicy policy = policy(false, 4);
        Fixture generic = fixture("llama", false, false, false, false, 30, 4096, 4096);

        assertFalse(policy.shouldUseMetalHalfMatvec(generic.traits(), generic.config(), 4));
    }

    @Test
    void gemma4AndPerLayerInputModelsAreAutoCandidates() throws Exception {
        DirectForwardMetalHalfMatvecAutoPolicy policy = policy(null, 4);
        Fixture gemma4 = fixture("gemma4_text", true, false, false, false, 16, 1024, 2048);
        Fixture perLayerInputs = fixture("custom", false, false, false, true, 16, 1024, 2048);

        assertTrue(policy.shouldUseMetalHalfMatvec(gemma4.traits(), gemma4.config(), 4));
        assertTrue(policy.shouldUseMetalHalfMatvec(perLayerInputs.traits(), perLayerInputs.config(), 4));
    }

    @Test
    void qwenAutoCandidateUsesSmallerModelShapeRules() throws Exception {
        DirectForwardMetalHalfMatvecAutoPolicy policy = policy(null, 4);
        Fixture qwen = fixture("qwen2", false, false, true, false, 20, 2048, 8192);
        Fixture tooWideQwen = fixture("qwen2", false, false, true, false, 20, 4096, 8192);
        Fixture tooShallowQwen = fixture("qwen2", false, false, true, false, 19, 2048, 8192);

        assertTrue(policy.shouldUseMetalHalfMatvec(qwen.traits(), qwen.config(), 4));
        assertFalse(policy.shouldUseMetalHalfMatvec(tooWideQwen.traits(), tooWideQwen.config(), 4));
        assertFalse(policy.shouldUseMetalHalfMatvec(tooShallowQwen.traits(), tooShallowQwen.config(), 4));
    }

    @Test
    void missingModelIdentityOrConfigIsNotAutoCandidate() throws Exception {
        DirectForwardMetalHalfMatvecAutoPolicy policy = policy(null, 4);
        Fixture blank = fixture("", false, false, false, false, 30, 4096, 4096);
        ModelConfigTraits traitsWithoutConfig =
                new ModelConfigTraits(null, "llama", 0, 0, false, false, false, false, false, false, false);

        assertFalse(policy.shouldUseMetalHalfMatvec(blank.traits(), blank.config(), 4));
        assertFalse(policy.shouldUseMetalHalfMatvec(traitsWithoutConfig, null, 4));
    }

    private static DirectForwardMetalHalfMatvecAutoPolicy policy(Boolean auto, int maxOutput) {
        return new DirectForwardMetalHalfMatvecAutoPolicy(auto, maxOutput);
    }

    private static ModelConfigTraits traits(
            String modelType,
            boolean nativeBf16Matvec,
            boolean requiresTurnAwarePromptBos,
            boolean preferLargeLogitsMatvec,
            boolean perLayerInputPath) {
        return new ModelConfigTraits(
                null, modelType, 0, 0, requiresTurnAwarePromptBos, preferLargeLogitsMatvec, perLayerInputPath, nativeBf16Matvec, false, false);
    }

    private static Fixture fixture(
            String modelType,
            boolean nativeBf16Matvec,
            boolean requiresTurnAwarePromptBos,
            boolean preferLargeLogitsMatvec,
            boolean perLayerInputPath,
            int layers,
            int hiddenSize,
            int intermediateSize) throws Exception {
        ModelConfig config = MAPPER.readValue("""
                {
                  "model_type": "%s",
                  "num_hidden_layers": %d,
                  "hidden_size": %d,
                  "intermediate_size": %d
                }
                """.formatted(modelType, layers, hiddenSize, intermediateSize), ModelConfig.class);
        ModelConfigTraits traits = new ModelConfigTraits(
                config,
                modelType,
                0,
                0,
                requiresTurnAwarePromptBos,
                preferLargeLogitsMatvec,
                perLayerInputPath,
                nativeBf16Matvec, false, false);
        return new Fixture(config, traits);
    }

    private record Fixture(ModelConfig config, ModelConfigTraits traits) {
    }
}
