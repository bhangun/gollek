/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.spi.model;

import tech.kayys.gollek.spi.model.ModelRuntimeTraits.PromptBosPolicy;

import java.util.Locale;
import java.util.Set;

/**
 * Prompt and tokenizer-control policy derived from model family traits.
 *
 * <p>This keeps prompt defaults, BOS insertion, and control-token validation
 * policy out of broader runtime traits so model-family prompt behavior can
 * evolve independently from attention and modality policy.
 */
public record ModelPromptTraits(
        PromptBosPolicy promptBosPolicy,
        Set<String> allowedControlTokenTexts,
        boolean validateContinuationTokensByDecode,
        boolean rejectEmptyDecodedTokens,
        boolean skipDefaultSystemPromptInjection,
        String defaultSystemPrompt) {

    public static final String DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";

    public ModelPromptTraits {
        promptBosPolicy = promptBosPolicy == null ? PromptBosPolicy.DEFAULT : promptBosPolicy;
        allowedControlTokenTexts = allowedControlTokenTexts == null
                ? Set.of()
                : Set.copyOf(allowedControlTokenTexts);
        defaultSystemPrompt = defaultSystemPrompt == null || defaultSystemPrompt.isBlank()
                ? DEFAULT_SYSTEM_PROMPT
                : defaultSystemPrompt;
    }
}
