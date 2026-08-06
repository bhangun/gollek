/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectGenerationRequestContextTest {

    @AfterEach
    void clearActiveProfile() {
        DirectInferenceProfiler.clearProfile();
    }

    @Test
    void createsSyncRequestContext() {
        DirectGenerationRequestContext context = DirectGenerationRequestContext.sync(null);

        assertFalse(context.requestId().isBlank());
        assertNotNull(context.startedAt());
        assertTrue(context.startedNanos() > 0L);
        assertEquals("sync", context.profile().mode);
        assertEquals("cpu", context.backend());
        assertNotNull(context.timings());
        assertEquals(0L, context.benchTimings().sessionAllocateNanos());
    }

    @Test
    void createsStreamRequestContext() {
        DirectGenerationRequestContext context = DirectGenerationRequestContext.stream(null);

        assertFalse(context.requestId().isBlank());
        assertEquals("stream", context.profile().mode);
        assertEquals("cpu", context.backend());
    }
}
