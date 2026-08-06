/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectConversationContinuationPlanTest {

    @Test
    void resolvesDeltaContinuation() {
        DirectConversationContinuationPlan plan =
                DirectConversationContinuationPlan.resolve(new long[] { 10, 20, 30, 40 }, 2, 2, null);

        assertFalse(plan.exactReplay());
        assertArrayEquals(new long[] { 30, 40 }, plan.deltaInputIds());
        assertEquals("conversation_delta", plan.promptTokenSource());

        Map<String, Object> metadata = plan.retainedKvMetadata();
        assertEquals(false, metadata.get("conversation_exact_replay"));
        assertEquals(true, metadata.get("conversation_delta_prefill"));
        assertEquals(2, metadata.get("conversation_cached_prefix_tokens"));
        assertEquals(2, metadata.get("conversation_delta_prompt_tokens"));
    }

    @Test
    void resolvesExactReplayContinuation() {
        DirectConversationContinuationPlan plan =
                DirectConversationContinuationPlan.resolve(new long[] { 10, 20 }, 2, 2, 99);

        assertTrue(plan.exactReplay());
        assertArrayEquals(new long[0], plan.deltaInputIds());
        assertEquals(99, plan.nextReplayTokenId());
        assertEquals("conversation_replay", plan.promptTokenSource());
    }

    @Test
    void rejectsExactReplayWithoutReplayToken() {
        assertThrows(IllegalArgumentException.class,
                () -> DirectConversationContinuationPlan.resolve(new long[] { 10, 20 }, 2, 2, null));
    }

    @Test
    void rejectsMismatchedCachePosition() {
        assertThrows(IllegalStateException.class,
                () -> DirectConversationContinuationPlan.resolve(new long[] { 10, 20, 30 }, 2, 1, null));
    }
}
