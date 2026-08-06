/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.sdk;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LegacyDirectTextPolicyTest {

    @TempDir
    Path tempDir;

    @Test
    void formatsPromptFromGemma4RuntimeTraitWithoutModelTypeString() {
        InferenceRequest request = InferenceRequest.builder()
                .model("local")
                .message(Message.user("where is jakarta"))
                .build();
        LegacyDirectModelProfile profile = new LegacyDirectModelProfile(
                null,
                "custom",
                ModelRuntimeTraits.builder()
                        .nativeBf16Matvec()
                        .build());

        String prompt = LegacyDirectTextPolicy.buildPrompt(request, request.getPrompt(), profile, null);

        assertTrue(prompt.startsWith("<bos><|turn>user\nwhere is jakarta"));
    }

    @Test
    void formatsPromptFromQwenRuntimeTraitWithoutModelTypeString() {
        InferenceRequest request = InferenceRequest.builder()
                .model("local")
                .messages(List.of(Message.system("be concise"), Message.user("where is jakarta")))
                .build();
        LegacyDirectModelProfile profile = new LegacyDirectModelProfile(
                null,
                "custom",
                ModelRuntimeTraits.builder()
                        .qwenText()
                        .build());

        String prompt = LegacyDirectTextPolicy.buildPrompt(request, request.getPrompt(), profile, null);

        assertTrue(prompt.startsWith("<|im_start|>system\nbe concise"));
        assertTrue(prompt.contains("<|im_start|>user\nwhere is jakarta"));
    }

    @Test
    void sanitizesGemma4ResponseFromRuntimeTrait() {
        InferenceResponse response = InferenceResponse.builder()
                .requestId("r")
                .content("<|channel>thought\nscratch<channel|>answer<turn|><|tool_response>")
                .build();
        LegacyDirectModelProfile profile = new LegacyDirectModelProfile(
                null,
                "custom",
                ModelRuntimeTraits.builder()
                        .nativeBf16Matvec()
                        .build());

        InferenceResponse sanitized = LegacyDirectTextPolicy.sanitizeResponse(response, profile);

        assertEquals("answer", sanitized.getContent());
    }

    @Test
    void loadedProfileResolvesRuntimeTraitsFromConfigFallback() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), "{\"model_type\":\"gemma4_text\"}");

        LegacyDirectModelProfile profile = LegacyDirectModelProfile.load(tempDir, null);

        assertTrue(profile.nativeBf16Matvec());
    }

    @Test
    void gemma3SafetyGateUsesRuntimeTraits() {
        LegacyDirectModelProfile profile = new LegacyDirectModelProfile(
                null,
                "custom",
                ModelRuntimeTraits.builder()
                        .gemma3Text()
                        .build());

        assertThrows(IllegalStateException.class,
                () -> LegacyDirectTextPolicy.validateDirectModelSupport(profile, "gollek.test.disabled"));
    }
}
