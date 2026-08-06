/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.sdk;

import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.engine.prompt.PromptTemplateCompat;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Pattern;

final class LegacyDirectTextPolicy {

    private static final Pattern GEMMA4_THOUGHT_CHANNEL =
            Pattern.compile("^<\\|channel>thought\\n.*?<channel\\|>", Pattern.DOTALL);
    private static final Pattern GEMMA4_GENERIC_CHANNEL_OPEN =
            Pattern.compile("^<\\|channel>[^\\n]*\\n", Pattern.DOTALL);

    private LegacyDirectTextPolicy() {
    }

    static String buildPrompt(
            InferenceRequest request,
            String rawPrompt,
            LegacyDirectModelProfile modelProfile,
            Logger log) {
        boolean useRawPrompt = shouldUseRawPrompt(request, rawPrompt, modelProfile);
        boolean shouldFormat = shouldFormatPrompt(modelProfile);
        if (Boolean.getBoolean("gollek.verbose")) {
            int messageCount = request == null || request.getMessages() == null ? -1 : request.getMessages().size();
            System.out.printf(
                    "[DEBUG-PROMPT] modelType=%s runtimeTraits=%s shouldFormat=%s useRaw=%s messageCount=%d%n",
                    modelProfile.modelType(), modelProfile.runtimeTraits(), shouldFormat, useRawPrompt, messageCount);
            System.out.flush();
        }
        if (useRawPrompt || !shouldFormat) {
            return rawPrompt;
        }

        List<Message> messages = request.getMessages();
        if (messages == null || messages.isEmpty()) {
            return rawPrompt;
        }

        try {
            return PromptTemplateCompat.format(
                    new ArrayList<>(messages),
                    modelProfile.modelType(),
                    modelProfile.runtimeTraits());
        } catch (Exception e) {
            if (log != null) {
                log.debugf("Falling back to raw prompt for legacy direct path: %s", e.getMessage());
            }
            return rawPrompt;
        }
    }

    static void validateDirectModelSupport(LegacyDirectModelProfile modelProfile, String overrideProperty) {
        if (!modelProfile.geluGatedFfn()) {
            return;
        }
        if (Boolean.getBoolean(overrideProperty)) {
            return;
        }
        throw new IllegalStateException(
                "Gemma3 direct safetensor path is temporarily disabled due incorrect generation quality. "
                        + "Use GGUF/LiteRT route, or override only for debugging with -D"
                        + overrideProperty + "=true");
    }

    static InferenceResponse sanitizeResponse(InferenceResponse response, LegacyDirectModelProfile modelProfile) {
        if (response == null || response.getContent() == null || !modelProfile.nativeBf16Matvec()) {
            return response;
        }

        String content = response.getContent();
        content = GEMMA4_THOUGHT_CHANNEL.matcher(content).replaceFirst("");
        content = GEMMA4_GENERIC_CHANNEL_OPEN.matcher(content).replaceFirst("");
        content = stripGemma4InlineMarkers(content);

        if (content.equals(response.getContent())) {
            return response;
        }
        return response.toBuilder().content(content).build();
    }

    static String sanitizeStreamingDelta(
            String content,
            LegacyDirectModelProfile modelProfile,
            AtomicBoolean leadingChannelsPending) {
        if (content == null || !modelProfile.nativeBf16Matvec()) {
            return content;
        }
        String text = content;
        if (leadingChannelsPending.get()) {
            text = GEMMA4_THOUGHT_CHANNEL.matcher(text).replaceFirst("");
            text = GEMMA4_GENERIC_CHANNEL_OPEN.matcher(text).replaceFirst("");
            if (!text.isBlank()) {
                leadingChannelsPending.set(false);
            }
        }
        return stripGemma4InlineMarkers(text);
    }

    private static boolean shouldUseRawPrompt(
            InferenceRequest request,
            String rawPrompt,
            LegacyDirectModelProfile modelProfile) {
        if (request == null) {
            return true;
        }
        if (modelProfile.geluGatedFfn() || modelProfile.nativeBf16Matvec()) {
            return false;
        }
        List<Message> messages = request.getMessages();
        if (messages == null || messages.size() != 1) {
            return false;
        }
        Message only = messages.getFirst();
        if (only.getRole() != Message.Role.USER) {
            return false;
        }
        String messagePrompt = only.getContent();
        if (messagePrompt == null) {
            return false;
        }
        return rawPrompt.equals(messagePrompt);
    }

    private static boolean shouldFormatPrompt(LegacyDirectModelProfile modelProfile) {
        return modelProfile.hasKnownPromptTemplate();
    }

    private static String stripGemma4InlineMarkers(String text) {
        if (text == null || text.isEmpty()) {
            return text;
        }
        String content = text.replace("<channel|>", "");
        content = content.replace("<turn|>", "");
        return content.replace("<|tool_response>", "");
    }
}
