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

class DirectForwardMetalHalfMatvecCorePolicyTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void defaultsAutoEnableGenericCandidateWithinThreshold() throws Exception {
        DirectForwardMetalHalfMatvecCoreOptions options =
                DirectForwardMetalHalfMatvecCoreOptions.defaults().withHalfMatvec(null, false, null, 4);
        DirectForwardMetalHalfMatvecCorePolicy policy = policy(options);
        Fixture generic = fixture("llama", false, false, false, 30, 4096, 4096);

        assertTrue(policy.shouldUseMetalHalfMatvec(generic.traits(), generic.config(), 4, "q_proj"));
        assertFalse(policy.shouldUseMetalHalfMatvec(generic.traits(), generic.config(), 5, "q_proj"));
    }

    @Test
    void disableAndAutoFlagsGateHalfMatvec() throws Exception {
        Fixture generic = fixture("llama", false, false, false, 30, 4096, 4096);

        assertFalse(policy(DirectForwardMetalHalfMatvecCoreOptions.defaults()
                .withHalfMatvec(null, true, null, 4))
                .shouldUseMetalHalfMatvec(generic.traits(), generic.config(), 4, "q_proj"));
        assertFalse(policy(DirectForwardMetalHalfMatvecCoreOptions.defaults()
                .withHalfMatvec(null, false, false, 4))
                .shouldUseMetalHalfMatvec(generic.traits(), generic.config(), 4, "q_proj"));
    }

    @Test
    void explicitHalfMatvecEnableIgnoresModelCandidateButHonorsThreshold() throws Exception {
        DirectForwardMetalHalfMatvecCoreOptions options =
                DirectForwardMetalHalfMatvecCoreOptions.defaults().withHalfMatvec(true, false, null, 4);
        DirectForwardMetalHalfMatvecCorePolicy policy = policy(options);
        Fixture unknown = fixture("", false, false, false, 32, 4096, 11008);

        assertTrue(policy.shouldUseMetalHalfMatvec(unknown.traits(), unknown.config(), 4, "q_proj"));
        assertFalse(policy.shouldUseMetalHalfMatvec(unknown.traits(), unknown.config(), 5, "q_proj"));
    }

    @Test
    void logitsOptionsApplyFamilySpecificThresholds() throws Exception {
        DirectForwardMetalHalfMatvecCoreOptions options =
                DirectForwardMetalHalfMatvecCoreOptions.defaults().withHalfMatvec(null, false, null, 99);
        DirectForwardMetalHalfMatvecLogitsOptions logits =
                DirectForwardMetalHalfMatvecLogitsOptions.defaults().withLogitsMaxOutputs(4, 5, 6);
        DirectForwardMetalHalfMatvecLogitsPolicy logitsPolicy =
                DirectForwardMetalHalfMatvecLogitsPolicy.from(logits);
        Fixture gemma4 = fixture("gemma4_text", true, false, false, 40, 3584, 14336);
        Fixture qwen = fixture("qwen2", false, false, true, 20, 2048, 8192);
        DirectForwardMetalHalfMatvecCorePolicy policy = policy(options, logitsPolicy);

        assertTrue(policy.shouldUseMetalHalfMatvec(gemma4.traits(), gemma4.config(), 4, "logits"));
        assertFalse(policy.shouldUseMetalHalfMatvec(gemma4.traits(), gemma4.config(), 5, "logits"));
        assertTrue(policy.shouldUseMetalHalfMatvec(qwen.traits(), qwen.config(), 6, "logits"));
        assertFalse(policy.shouldUseMetalHalfMatvec(qwen.traits(), qwen.config(), 7, "logits"));
    }

    @Test
    void turnAwarePromptBosLogitsMatvecCanBypassAutoWhenExplicit() throws Exception {
        DirectForwardMetalHalfMatvecCoreOptions options =
                DirectForwardMetalHalfMatvecCoreOptions.defaults().withHalfMatvec(null, false, false, 4);
        DirectForwardMetalHalfMatvecLogitsOptions logits =
                DirectForwardMetalHalfMatvecLogitsOptions.defaults()
                        .withLogitsMaxOutputs(4, 4, 4)
                        .withGemma3Logits(true, false);
        DirectForwardMetalHalfMatvecLogitsPolicy logitsPolicy =
                DirectForwardMetalHalfMatvecLogitsPolicy.from(logits);
        Fixture gemma3 = fixture("gemma3_text", false, true, false, 40, 3584, 14336);
        DirectForwardMetalHalfMatvecCorePolicy policy = policy(options, logitsPolicy);

        assertTrue(policy.shouldUseMetalHalfMatvec(gemma3.traits(), gemma3.config(), 4, "logits"));
        assertFalse(policy.shouldUseMetalHalfMatvec(gemma3.traits(), gemma3.config(), 4, "q_proj"));
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

    private static DirectForwardMetalHalfMatvecLogitsPolicy defaultLogitsPolicy() {
        return DirectForwardMetalHalfMatvecLogitsPolicy.from(DirectForwardMetalHalfMatvecLogitsOptions.defaults());
    }

    private static DirectForwardMetalHalfMatvecCorePolicy policy(
            DirectForwardMetalHalfMatvecCoreOptions options) {
        return policy(options, defaultLogitsPolicy());
    }

    private static DirectForwardMetalHalfMatvecCorePolicy policy(
            DirectForwardMetalHalfMatvecCoreOptions options,
            DirectForwardMetalHalfMatvecLogitsPolicy logitsPolicy) {
        return DirectForwardMetalHalfMatvecCorePolicy.from(options, logitsPolicy);
    }
}
