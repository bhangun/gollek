/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.safetensor.engine.prompt.PromptTemplateCompat;
import tech.kayys.gollek.spi.Message;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Shapes provider messages into engine-ready prompt artifacts.
 */
@ApplicationScoped
public class InferencePromptPlanner {

    InferencePromptPlan plan(
            List<Message> messages,
            String modelType,
            ModelRuntimeTraits runtimeTraits,
            boolean suppressDefaultSystemPrompt) {
        Objects.requireNonNull(messages, "messages");
        ModelRuntimeTraits traits = runtimeTraits == null ? ModelRuntimeTraits.EMPTY : runtimeTraits;
        String resolvedModelType = modelType == null ? "" : modelType;

        List<Message> shapedMessages = new ArrayList<>(messages);
        boolean hasSystem = shapedMessages.stream()
                .anyMatch(message -> message.getRole() == Message.Role.SYSTEM);
        boolean injectDefaultSystem = !hasSystem
                && !suppressDefaultSystemPrompt
                && !traits.skipDefaultSystemPromptInjection();
        if (injectDefaultSystem) {
            shapedMessages.add(0, Message.system(traits.defaultSystemPrompt()));
        }

        String formattedPrompt = PromptTemplateCompat.format(shapedMessages, resolvedModelType, traits);
        PreparedPrompt preparedPrompt = PreparedPrompt.of(formattedPrompt, injectDefaultSystem, resolvedModelType);
        return new InferencePromptPlan(preparedPrompt, ttsPrompt(shapedMessages, formattedPrompt));
    }

    private static String ttsPrompt(List<Message> messages, String fallbackPrompt) {
        return messages.stream()
                .filter(message -> message.getRole() == Message.Role.USER)
                .map(Message::getContent)
                .reduce((first, second) -> second)
                .orElse(fallbackPrompt);
    }
}
