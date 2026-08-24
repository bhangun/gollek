/*
 * Gollek CLI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.sdk.route.RunnerRouteBenchmarkCache;
import tech.kayys.gollek.sdk.route.RunnerRoutePolicy;
import tech.kayys.gollek.sdk.route.RunnerRouteReportContract;
import tech.kayys.gollek.safetensor.engine.route.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectSafetensorTextPolicyTest {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @Test
    void gemma4DefaultRepeatPenaltyFallsBackToNeutral() {
        DirectSafetensorRunProfile profile = profile("gemma4", ModelRuntimeTraits.builder()
                .nativeBf16Matvec()
                .build());

        assertEquals(1.0f, DirectSafetensorTextPolicy.normalizeRepeatPenalty(profile, 1.1d));
        assertEquals(1.2f, DirectSafetensorTextPolicy.normalizeRepeatPenalty(profile, 1.2d));
    }

    @Test
    void qwenProfileUsesChatTemplateFormatterWithRuntimeTraits() {
        DirectSafetensorRunProfile profile = profile("qwen", ModelRuntimeTraits.builder()
                .build());

        String prepared = DirectSafetensorTextPolicy.preparePrompt(profile, "You are concise.", "where is jakarta");

        assertTrue(prepared.startsWith("<|im_start|>system"));
        assertTrue(prepared.contains("You are concise."));
        assertTrue(prepared.contains("<|im_start|>user"));
        assertTrue(prepared.contains("where is jakarta"));
    }

    @Test
    void guardedGemma4UnifiedProfileUsesGemma4ChatTemplate() throws Exception {
        DirectSafetensorRunProfile profile = gemma4UnifiedProfile();

        String prepared = DirectSafetensorTextPolicy.preparePrompt(profile, null, "where is jakarta");

        assertEquals("<bos><|turn>user\nwhere is jakarta<turn|>\n<|turn>model\n", prepared);
    }

    @Test
    void unguardedGemma4TextProfileKeepsRawPrompt() {
        DirectSafetensorRunProfile profile = profile("gemma4_text", ModelRuntimeTraits.builder()
                .nativeBf16Matvec()
                .build());

        String prompt = "where is jakarta";

        assertSame(prompt, DirectSafetensorTextPolicy.preparePrompt(profile, null, prompt));
    }

    @Test
    void preformattedPromptsStayUntouched() {
        DirectSafetensorRunProfile profile = profile("qwen2", ModelRuntimeTraits.builder()
                .build());
        String prompt = "<|im_start|>user\nwhere is jakarta<|im_end|>";

        assertSame(prompt, DirectSafetensorTextPolicy.preparePrompt(profile, "ignored", prompt));
    }

    @Test
    void gemma4ChannelMarkupIsSanitized() {
        DirectSafetensorRunProfile profile = profile("gemma4", ModelRuntimeTraits.builder()
                .nativeBf16Matvec()
                .build());
        InferenceResponse response = InferenceResponse.builder()
                .requestId("req-test")
                .content("<|channel>thought\nscratch<channel|>Jakarta is in Indonesia.")
                .build();

        InferenceResponse sanitized = DirectSafetensorTextPolicy.sanitizeResponse(response, profile);

        assertEquals("Jakarta is in Indonesia.", sanitized.getContent());
    }

    private static DirectSafetensorRunProfile profile(String modelType, ModelRuntimeTraits traits) {
        return new DirectSafetensorRunProfile(null, modelType, traits);
    }

    private static DirectSafetensorRunProfile gemma4UnifiedProfile() throws Exception {
        ModelConfig config = OBJECT_MAPPER.readValue("""
                {
                  "model_type": "gemma4_unified",
                  "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                  "text_config": {
                    "model_type": "gemma4_text"
                  }
                }
                """, ModelConfig.class);
        return new DirectSafetensorRunProfile(config, "gemma4_unified", ModelRuntimeTraits.builder()
                .nativeBf16Matvec()
                .build());
    }
}
