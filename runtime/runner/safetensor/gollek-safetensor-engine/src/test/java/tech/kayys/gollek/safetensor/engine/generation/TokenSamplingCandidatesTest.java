/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TokenSamplingCandidatesTest {

    @Test
    void selectsAllCandidatesWhenTopKDisabled() {
        float[] logits = { 1.0f, 3.0f, 2.0f };
        int[] indices = new int[logits.length];

        TokenSamplingCandidates.Prepared candidates = TokenSamplingCandidates.prepare(logits, 0, indices);

        assertEquals(3, candidates.limit());
        assertEquals(1, candidates.indices()[0]);
        assertEquals(2, candidates.indices()[1]);
        assertEquals(0, candidates.indices()[2]);
    }

    @Test
    void selectsOnlyTopKCandidatesInDescendingLogitOrder() {
        float[] logits = { 1.0f, 10.0f, 3.0f, 7.0f };
        int[] indices = new int[logits.length];

        TokenSamplingCandidates.Prepared candidates = TokenSamplingCandidates.prepare(logits, 2, indices);

        assertEquals(2, candidates.limit());
        assertEquals(1, candidates.indices()[0]);
        assertEquals(3, candidates.indices()[1]);
    }

    @Test
    void handlesSingleCandidateWithoutSortingStackGrowth() {
        float[] logits = { 5.0f };
        int[] indices = new int[logits.length];

        TokenSamplingCandidates.Prepared candidates = TokenSamplingCandidates.prepare(logits, 0, indices);

        assertEquals(1, candidates.limit());
        assertEquals(0, candidates.indices()[0]);
    }
}
