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

class DirectForwardMetalHalfMatvecRoutingPolicyTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void defaultsAutoEnableGenericCandidateWithinThreshold() throws Exception {
        DirectForwardMetalHalfMatvecRoutingPolicy policy = policy(
                DirectForwardMetalHalfMatvecOptions.defaults().withHalfMatvec(null, false, null, 4));
        Fixture generic = fixture("llama", false, false, false, 30, 4096, 4096);

        assertTrue(policy.shouldUseMetalHalfMatvec(generic.traits(), generic.config(), 4, "q_proj"));
        assertFalse(policy.shouldUseMetalHalfMatvec(generic.traits(), generic.config(), 5, "q_proj"));
    }

    @Test
    void disableAndAutoFlagsGateHalfMatvec() throws Exception {
        Fixture generic = fixture("llama", false, false, false, 30, 4096, 4096);

        assertFalse(policy(DirectForwardMetalHalfMatvecOptions.defaults()
                .withHalfMatvec(null, true, null, 4))
                .shouldUseMetalHalfMatvec(generic.traits(), generic.config(), 4, "q_proj"));
        assertFalse(policy(DirectForwardMetalHalfMatvecOptions.defaults()
                .withHalfMatvec(null, false, false, 4))
                .shouldUseMetalHalfMatvec(generic.traits(), generic.config(), 4, "q_proj"));
    }

    @Test
    void explicitHalfMatvecEnableIgnoresModelCandidateButHonorsThreshold() throws Exception {
        DirectForwardMetalHalfMatvecRoutingPolicy policy = policy(
                DirectForwardMetalHalfMatvecOptions.defaults().withHalfMatvec(true, false, null, 4));
        Fixture unknown = fixture("", false, false, false, 32, 4096, 11008);

        assertTrue(policy.shouldUseMetalHalfMatvec(unknown.traits(), unknown.config(), 4, "q_proj"));
        assertFalse(policy.shouldUseMetalHalfMatvec(unknown.traits(), unknown.config(), 5, "q_proj"));
    }

    @Test
    void pairPoliciesKeepPairSpecificOverridesSeparate() throws Exception {
        Fixture generic = fixture("llama", false, false, false, 30, 4096, 4096);
        Fixture gemma4 = fixture("gemma4_text", true, false, false, 40, 3584, 14336);

        assertTrue(policy(DirectForwardMetalHalfMatvecOptions.defaults()
                .withHalfMatvecPair(true, false))
                .shouldUseMetalHalfMatvecPair(generic.traits(), generic.config(), 4));
        assertFalse(policy(DirectForwardMetalHalfMatvecOptions.defaults()
                .withHalfMatvecPair(true, true))
                .shouldUseMetalHalfMatvecPair(generic.traits(), generic.config(), 4));
        assertFalse(policy(DirectForwardMetalHalfMatvecOptions.defaults())
                .shouldUseMetalHalfLinearPair(gemma4.traits(), false, true));
        assertTrue(policy(DirectForwardMetalHalfMatvecOptions.defaults())
                .shouldUseMetalHalfLinearPair(gemma4.traits(), true, true));
        assertFalse(policy(DirectForwardMetalHalfMatvecOptions.defaults()
                .withHalfLinearPair(false, false))
                .shouldUseMetalHalfLinearPair(generic.traits(), true, false));
    }

    @Test
    void logitsMpsMatvecRequiresExplicitEnableAndRejectsNativeBf16() throws Exception {
        DirectForwardMetalHalfMatvecRoutingPolicy policy = policy(
                DirectForwardMetalHalfMatvecOptions.defaults().withLogitsMpsMatvec(true, false, 10, 4));
        Fixture generic = fixture("llama", false, false, false, 30, 4096, 4096);
        Fixture gemma4 = fixture("gemma4_text", true, false, false, 40, 3584, 14336);

        assertTrue(policy.shouldUseMetalLogitsMpsMatvec(generic.traits(), 10, 4, "logits"));
        assertFalse(policy.shouldUseMetalLogitsMpsMatvec(generic.traits(), 9, 4, "logits"));
        assertFalse(policy.shouldUseMetalLogitsMpsMatvec(generic.traits(), 10, 5, "logits"));
        assertFalse(policy.shouldUseMetalLogitsMpsMatvec(generic.traits(), 10, 4, "q_proj"));
        assertFalse(policy.shouldUseMetalLogitsMpsMatvec(gemma4.traits(), 10, 4, "logits"));
    }

    @Test
    void familySpecificLogitsThresholdsStayScoped() throws Exception {
        DirectForwardMetalHalfMatvecRoutingPolicy policy = policy(
                DirectForwardMetalHalfMatvecOptions.defaults().withLogitsMaxOutputs(4, 5, 6));
        Fixture gemma4 = fixture("gemma4_text", true, false, false, 40, 3584, 14336);
        Fixture qwen = fixture("qwen2", false, false, true, 20, 2048, 8192);

        assertTrue(policy.shouldUseMetalHalfMatvec(gemma4.traits(), gemma4.config(), 4, "logits"));
        assertFalse(policy.shouldUseMetalHalfMatvec(gemma4.traits(), gemma4.config(), 5, "logits"));
        assertTrue(policy.shouldUseMetalHalfMatvec(qwen.traits(), qwen.config(), 6, "logits"));
        assertFalse(policy.shouldUseMetalHalfMatvec(qwen.traits(), qwen.config(), 7, "logits"));
    }

    @Test
    void turnAwarePromptBosLogitsMatvecRequiresExplicitEnable() throws Exception {
        Fixture gemma3 = fixture("gemma3_text", false, true, false, 40, 3584, 14336);

        assertFalse(policy(DirectForwardMetalHalfMatvecOptions.defaults()
                .withHalfMatvec(null, false, false, 4)
                .withLogitsMaxOutputs(4, 4, 4))
                .shouldUseMetalHalfMatvec(gemma3.traits(), gemma3.config(), 4, "logits"));
        assertTrue(policy(DirectForwardMetalHalfMatvecOptions.defaults()
                .withHalfMatvec(null, false, false, 4)
                .withLogitsMaxOutputs(4, 4, 4)
                .withGemma3Logits(true, false))
                .shouldUseMetalHalfMatvec(gemma3.traits(), gemma3.config(), 4, "logits"));
    }

    @Test
    void transposedMatvecDefaultsToNativeBf16LogitsOnly() throws Exception {
        DirectForwardMetalHalfMatvecRoutingPolicy policy = policy(
                DirectForwardMetalHalfMatvecOptions.defaults().withTransposedHalfMatvec(null, false, false, 4));
        Fixture generic = fixture("llama", false, false, false, 30, 4096, 4096);
        Fixture gemma4 = fixture("gemma4_text", true, false, false, 40, 3584, 14336);

        assertTrue(policy.shouldUseMetalTransposedHalfMatvec(gemma4.traits(), 4, "logits"));
        assertFalse(policy.shouldUseMetalTransposedHalfMatvec(gemma4.traits(), 5, "logits"));
        assertFalse(policy.shouldUseMetalTransposedHalfMatvec(generic.traits(), 4, "logits"));
        assertFalse(policy.shouldUseMetalTransposedHalfMatvec(gemma4.traits(), 4, "q_proj"));
    }

    @Test
    void transposedMatvecRequiresExplicitAllForHiddenProjectors() throws Exception {
        Fixture generic = fixture("llama", false, false, false, 30, 4096, 4096);

        assertFalse(policy(DirectForwardMetalHalfMatvecOptions.defaults()
                .withTransposedHalfMatvec(true, false, false, 4))
                .shouldUseMetalTransposedHalfMatvec(generic.traits(), 4, "q_proj"));
        assertTrue(policy(DirectForwardMetalHalfMatvecOptions.defaults()
                .withTransposedHalfMatvec(true, true, false, 4))
                .shouldUseMetalTransposedHalfMatvec(generic.traits(), 4, "q_proj"));
    }

    private static DirectForwardMetalHalfMatvecRoutingPolicy policy(
            DirectForwardMetalHalfMatvecOptions options) {
        return DirectForwardMetalHalfMatvecRoutingPolicy.from(options);
    }

    private static Fixture fixture(String modelType, boolean nativeBf16Matvec, boolean gemma3Text, boolean qwenText,
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
                new ModelConfigTraits(config, modelType, 0, 0, nativeBf16Matvec, gemma3Text, qwenText, false, false, false, false);
        return new Fixture(config, traits);
    }

    private record Fixture(ModelConfig config, ModelConfigTraits traits) {
    }
}
