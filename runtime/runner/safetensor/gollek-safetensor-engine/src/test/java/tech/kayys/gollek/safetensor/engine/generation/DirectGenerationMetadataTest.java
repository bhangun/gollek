/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectGenerationMetadataTest {

    @Test
    void buildsPretokenizedPromptMetadata() {
        Map<String, Object> metadata = DirectGenerationMetadata.pretokenizedPrompt();

        assertEquals("pretokenized", metadata.get("prompt_token_source"));
        assertEquals(1, metadata.size());
    }

    @Test
    void buildsRetainedPretokenizedConversationMetadata() {
        Map<String, Object> metadata = DirectGenerationMetadata.retainedPretokenizedConversation();

        assertEquals("pretokenized", metadata.get("prompt_token_source"));
        assertEquals(true, metadata.get("conversation_kv_retained"));
    }

    @Test
    void buildsContinuationMetadataForDeltaAndExactReplay() {
        Map<String, Object> delta = DirectGenerationMetadata.retainedContinuation(false, 4, 2);
        assertEquals("conversation_delta", delta.get("prompt_token_source"));
        assertEquals(4, delta.get("conversation_cached_prefix_tokens"));
        assertEquals(2, delta.get("conversation_delta_prompt_tokens"));
        assertTrue((Boolean) delta.get("conversation_delta_prefill"));
        assertFalse((Boolean) delta.get("conversation_exact_replay"));

        Map<String, Object> replay = DirectGenerationMetadata.retainedContinuation(true, 4, 0);
        assertEquals("conversation_replay", replay.get("prompt_token_source"));
        assertFalse((Boolean) replay.get("conversation_delta_prefill"));
        assertTrue((Boolean) replay.get("conversation_exact_replay"));
    }
}
