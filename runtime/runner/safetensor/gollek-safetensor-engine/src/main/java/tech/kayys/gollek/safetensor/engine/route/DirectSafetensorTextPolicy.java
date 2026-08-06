/*
 * Gollek CLI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.route;

import tech.kayys.gollek.models.core.ChatTemplateFormatter;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.util.List;
import java.util.regex.Pattern;

public final class DirectSafetensorTextPolicy {

    private static final Pattern GEMMA4_THOUGHT_CHANNEL =
            Pattern.compile("^<\\|channel>thought\\n.*?<channel\\|>", Pattern.DOTALL);
    private static final Pattern GEMMA4_GENERIC_CHANNEL_OPEN =
            Pattern.compile("^<\\|channel>[^\\n]*\\n", Pattern.DOTALL);

    private DirectSafetensorTextPolicy() {
    }

    public static float normalizeRepeatPenalty(DirectSafetensorRunProfile profile, double requestedRepeatPenalty) {
        if (profile.nativeBf16Matvec() && Math.abs(requestedRepeatPenalty - 1.1d) < 1.0e-6) {
            return 1.0f;
        }
        return (float) requestedRepeatPenalty;
    }

    public static String preparePrompt(DirectSafetensorRunProfile profile, String systemPrompt, String userPrompt) {
        if (userPrompt == null || userPrompt.isBlank()) {
            return userPrompt;
        }
        if (looksLikePreformattedPrompt(userPrompt)) {
            return userPrompt;
        }
        if (!shouldFormatPrompt(profile)) {
            return userPrompt;
        }
        try {
            if (systemPrompt != null && !systemPrompt.isBlank()) {
                return ChatTemplateFormatter.format(
                        List.of(Message.system(systemPrompt), Message.user(userPrompt)),
                        profile.modelType());
            }
            return ChatTemplateFormatter.format(
                    List.of(Message.user(userPrompt)),
                    profile.modelType());
        } catch (Exception ignored) {
            return userPrompt;
        }
    }

    public static InferenceResponse sanitizeResponse(InferenceResponse response, DirectSafetensorRunProfile profile) {
        if (response == null || response.getContent() == null) {
            return response;
        }
        if (!profile.nativeBf16Matvec()) {
            return response;
        }

        String content = response.getContent();
        content = GEMMA4_THOUGHT_CHANNEL.matcher(content).replaceFirst("");
        content = GEMMA4_GENERIC_CHANNEL_OPEN.matcher(content).replaceFirst("");
        content = content.replace("<channel|>", "");
        content = content.replace("<turn|>", "");
        content = content.replace("<|tool_response>", "");

        if (content.equals(response.getContent())) {
            return response;
        }
        return response.toBuilder().content(content).build();
    }

    private static boolean shouldFormatPrompt(DirectSafetensorRunProfile profile) {
        if (profile.modelType().isBlank()) {
            return false;
        }
        if (profile.nativeBf16Matvec()) {
            return profile.unifiedMultimodal();
        }
        if (profile.requiresChatTemplateFormatting()) {
            return true;
        }
        return ChatTemplateFormatter.supportsModelType(profile.modelType());
    }

    private static boolean looksLikePreformattedPrompt(String prompt) {
        if (prompt == null) {
            return false;
        }
        String trimmed = prompt.stripLeading();
        if (trimmed.startsWith("<bos>")) {
            trimmed = trimmed.substring("<bos>".length()).stripLeading();
        }
        return trimmed.startsWith("<|turn>")
                || trimmed.startsWith("<start_of_turn>")
                || trimmed.startsWith("<|im_start|>")
                || trimmed.startsWith("<|begin_of_text|>")
                || trimmed.startsWith("[INST]");
    }
}
