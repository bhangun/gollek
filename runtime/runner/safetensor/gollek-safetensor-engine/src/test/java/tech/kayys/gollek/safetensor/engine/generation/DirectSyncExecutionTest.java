/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DirectSyncExecutionTest {
    private static final Duration TIMEOUT = Duration.ofSeconds(2);

    @AfterEach
    void clearActiveProfile() {
        DirectInferenceProfiler.clearProfile();
    }

    @Test
    void createsSyncRequestContextAndClearsProfileAfterSuccess() {
        AtomicReference<InferenceProfile> profile = new AtomicReference<>();

        String requestId = DirectSyncExecution.create(null, null, "sync failed", "Direct sync failed", request -> {
            assertEquals("sync", request.profile().mode);
            assertEquals("cpu", request.backend());
            DirectInferenceProfiler.recordModelLoadNanos(7L);
            profile.set(request.profile());
            return request.requestId();
        }).await().atMost(TIMEOUT);

        assertFalse(requestId.isBlank());
        assertNotNull(profile.get());
        assertEquals(7L, profile.get().modelLoadNanos);

        DirectInferenceProfiler.recordModelLoadNanos(11L);

        assertEquals(7L, profile.get().modelLoadNanos);
    }

    @Test
    void wrapsFailureAndClearsProfile() {
        AtomicReference<InferenceProfile> profile = new AtomicReference<>();
        IllegalArgumentException cause = new IllegalArgumentException("boom");

        RuntimeException thrown = assertThrows(RuntimeException.class,
                () -> DirectSyncExecution.create(null, null, "sync failed", "Direct sync failed", request -> {
                    DirectInferenceProfiler.recordModelLoadNanos(3L);
                    profile.set(request.profile());
                    throw cause;
                }).await().atMost(TIMEOUT));

        assertEquals("Direct sync failed: boom", thrown.getMessage());
        assertSame(cause, thrown.getCause());

        DirectInferenceProfiler.recordModelLoadNanos(11L);

        assertEquals(3L, profile.get().modelLoadNanos);
    }

    @Test
    void requiresWork() {
        assertThrows(NullPointerException.class,
                () -> DirectSyncExecution.create(null, null, "sync failed", "Direct sync failed", null));
    }
}
