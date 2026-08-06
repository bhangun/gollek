/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.util.Map;

/**
 * Shared response metadata keys and common metadata shapes for direct generation.
 */
final class DirectGenerationMetadata {
    static final String PROMPT_TOKEN_SOURCE = "prompt_token_source";
    static final String PRETOKENIZED = "pretokenized";
    static final String CONVERSATION_DELTA = "conversation_delta";
    static final String CONVERSATION_REPLAY = "conversation_replay";

    private static final String CONVERSATION_KV_RETAINED = "conversation_kv_retained";
    private static final String CONVERSATION_DELTA_PREFILL = "conversation_delta_prefill";
    private static final String CONVERSATION_EXACT_REPLAY = "conversation_exact_replay";
    private static final String CONVERSATION_CACHED_PREFIX_TOKENS = "conversation_cached_prefix_tokens";
    private static final String CONVERSATION_DELTA_PROMPT_TOKENS = "conversation_delta_prompt_tokens";

    private DirectGenerationMetadata() {
    }

    static Map<String, Object> pretokenizedPrompt() {
        return Map.of(PROMPT_TOKEN_SOURCE, PRETOKENIZED);
    }

    static Map<String, Object> retainedPretokenizedConversation() {
        return Map.of(
                PROMPT_TOKEN_SOURCE, PRETOKENIZED,
                CONVERSATION_KV_RETAINED, true);
    }

    static String continuationPromptTokenSource(boolean exactReplay) {
        return exactReplay ? CONVERSATION_REPLAY : CONVERSATION_DELTA;
    }

    static Map<String, Object> retainedContinuation(
            boolean exactReplay,
            int cachedPrefixTokens,
            int deltaPromptTokens) {
        return Map.of(
                PROMPT_TOKEN_SOURCE, continuationPromptTokenSource(exactReplay),
                CONVERSATION_DELTA_PREFILL, !exactReplay,
                CONVERSATION_EXACT_REPLAY, exactReplay,
                CONVERSATION_CACHED_PREFIX_TOKENS, cachedPrefixTokens,
                CONVERSATION_DELTA_PROMPT_TOKENS, deltaPromptTokens,
                CONVERSATION_KV_RETAINED, true);
    }
}
