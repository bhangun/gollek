/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

/**
 * Prepared prompt state needed by direct generation entry points.
 */
record DirectPromptPreparation(
        Tokenizer tokenizer,
        ModelConfig config,
        DirectPromptTokens tokens) {

    static DirectPromptPreparation text(DirectLoadedModel model, String prompt, InferenceProfile profile,
            DebugOptions debug) {
        return text(model.tokenizer(), model.config(), model.runtimeTraits(), prompt, profile, debug);
    }

    static DirectPromptPreparation text(Tokenizer tokenizer, ModelConfig config, ModelRuntimeTraits traits,
            String prompt, InferenceProfile profile, DebugOptions debug) {
        DebugOptions effectiveDebug = DebugOptions.orNone(debug);
        if (effectiveDebug.enabled()) {
            System.out.printf("%s 3: get tokenizer/config%n", effectiveDebug.prefix());
            System.out.flush();
        }
        if (effectiveDebug.enabled()) {
            System.out.printf("%s 4: architecture cache%n", effectiveDebug.prefix());
            System.out.flush();
        }
        if (effectiveDebug.enabled()) {
            System.out.printf("%s 5: tokenize%n", effectiveDebug.prefix());
            if (effectiveDebug.printPromptText()) {
                System.out.printf("[DEBUG-PROMPT-TEXT] %s%n", DirectPromptTokens.printableText(prompt));
            }
            System.out.flush();
        }
        DirectPromptTokens tokens = DirectPromptTokens.encode(tokenizer, config, traits, prompt, profile);
        if (effectiveDebug.enabled()) {
            System.out.printf("%s 6: tokens=%d%n", effectiveDebug.prefix(), tokens.length());
            tokens.debugSequence(tokenizer, "prompt");
            System.out.flush();
        }
        return new DirectPromptPreparation(tokenizer, config, tokens);
    }

    static DirectPromptPreparation pretokenized(DirectLoadedModel model, long[] inputIds, DebugOptions debug) {
        return pretokenized(model.tokenizer(), model.config(), inputIds, debug);
    }

    static DirectPromptPreparation pretokenized(Tokenizer tokenizer, ModelConfig config, long[] inputIds,
            DebugOptions debug) {
        DirectPromptTokens tokens = DirectPromptTokens.of(inputIds);
        DebugOptions effectiveDebug = DebugOptions.orNone(debug);
        if (effectiveDebug.enabled() && !effectiveDebug.pretokenizedLabel().isBlank()) {
            System.out.printf("%s tokens=%d%n", effectiveDebug.pretokenizedLabel(), tokens.length());
            System.out.flush();
        }
        return new DirectPromptPreparation(tokenizer, config, tokens);
    }

    long[] ids() {
        return tokens.ids();
    }

    int length() {
        return tokens.length();
    }

    record DebugOptions(boolean enabled, String prefix, boolean printPromptText, String pretokenizedLabel) {
        private static final DebugOptions NONE = new DebugOptions(false, "", false, "");

        static DebugOptions none() {
            return NONE;
        }

        static DebugOptions text(boolean enabled, String prefix, boolean printPromptText) {
            return new DebugOptions(enabled, prefix == null ? "" : prefix, printPromptText, "");
        }

        static DebugOptions pretokenized(boolean enabled, String label) {
            return new DebugOptions(enabled, "", false, label == null ? "" : label);
        }

        private static DebugOptions orNone(DebugOptions debug) {
            return debug == null ? NONE : debug;
        }
    }
}
