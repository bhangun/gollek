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
 * Validated prompt-token input for direct generation entry points.
 */
record DirectPromptTokens(long[] ids) {

    DirectPromptTokens {
        if (ids == null) {
            throw new IllegalArgumentException("Prompt token ids must not be null");
        }
        if (ids.length == 0) {
            throw new IllegalArgumentException("Prompt resulted in zero tokens. Please provide a valid prompt.");
        }
    }

    static DirectPromptTokens of(long[] ids) {
        return new DirectPromptTokens(ids);
    }

    static DirectPromptTokens encode(Tokenizer tokenizer, ModelConfig config, ModelRuntimeTraits traits,
            String prompt, InferenceProfile profile) {
        return of(encodeIds(tokenizer, config, traits, prompt, profile));
    }

    static long[] encodeIds(Tokenizer tokenizer, ModelConfig config, ModelRuntimeTraits traits,
            String prompt, InferenceProfile profile) {
        long tTokenize0 = System.nanoTime();
        long[] ids = tokenizer.encode(prompt, GenerationTokenPolicy.encodeOptionsFor(config, traits, prompt));
        if (profile != null) {
            profile.tokenizeNanos += System.nanoTime() - tTokenize0;
        }
        return ids;
    }

    int length() {
        return ids.length;
    }

    void debugSequence(Tokenizer tokenizer, String label) {
        GenerationTokenDebug.tokenSequence(tokenizer, ids, label);
    }

    static String printableText(String text) {
        return GenerationTokenDebug.printableText(text);
    }
}
