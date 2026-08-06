/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.util.function.BiConsumer;

/**
 * Centralizes direct-generation verbose debug behavior.
 */
final class DirectGenerationDebug {
    private static final String VERBOSE_PROPERTY = "gollek.verbose";
    private static final String TEXT_PREFIX = "[DEBUG]";
    private static final String STREAM_PREFIX = "[DEBUG-S]";

    private final boolean enabled;
    private final String prefix;

    private DirectGenerationDebug(boolean enabled, String prefix) {
        this.enabled = enabled;
        this.prefix = prefix == null || prefix.isBlank() ? TEXT_PREFIX : prefix;
    }

    static DirectGenerationDebug text() {
        return of(Boolean.getBoolean(VERBOSE_PROPERTY), TEXT_PREFIX);
    }

    static DirectGenerationDebug stream() {
        return of(Boolean.getBoolean(VERBOSE_PROPERTY), STREAM_PREFIX);
    }

    static DirectGenerationDebug of(boolean enabled, String prefix) {
        return new DirectGenerationDebug(enabled, prefix);
    }

    boolean enabled() {
        return enabled;
    }

    String prefix() {
        return prefix;
    }

    DirectPromptPreparation.DebugOptions textPrompt(boolean printPromptText) {
        return DirectPromptPreparation.DebugOptions.text(enabled, prefix, printPromptText);
    }

    DirectPromptPreparation.DebugOptions pretokenized(String label) {
        return DirectPromptPreparation.DebugOptions.pretokenized(enabled, label);
    }

    void step(int step, String message) {
        if (!enabled) {
            return;
        }
        System.out.printf("%s %d: %s%n", prefix, step, message);
        System.out.flush();
    }

    void chosenToken(Tokenizer tokenizer, int tokenId, int step) {
        if (enabled) {
            GenerationTokenDebug.chosenToken(tokenizer, tokenId, step);
        }
    }

    BiConsumer<Integer, Integer> chosenTokenObserver(Tokenizer tokenizer) {
        if (!enabled) {
            return null;
        }
        return (tokenId, step) -> GenerationTokenDebug.chosenToken(tokenizer, tokenId, step);
    }
}
