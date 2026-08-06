/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DirectGenerationFailuresTest {

    @Test
    void wrapsCauseWithGenerationPrefixAndDetail() {
        IllegalArgumentException cause = new IllegalArgumentException("bad shape");

        RuntimeException wrapped = DirectGenerationFailures.wrap(
                null, "log message", "Direct generation failed", cause);

        assertEquals("Direct generation failed: bad shape", wrapped.getMessage());
        assertSame(cause, wrapped.getCause());
    }

    @Test
    void omitsBlankCauseDetail() {
        IllegalStateException cause = new IllegalStateException("");

        RuntimeException wrapped = DirectGenerationFailures.wrap(
                null, "log message", "Direct conversation generation failed", cause);

        assertEquals("Direct conversation generation failed", wrapped.getMessage());
        assertSame(cause, wrapped.getCause());
    }

    @Test
    void defaultsBlankFailurePrefix() {
        RuntimeException wrapped = DirectGenerationFailures.wrap(
                null, "", "", new RuntimeException("boom"));

        assertEquals("Direct generation failed: boom", wrapped.getMessage());
    }

    @Test
    void requiresCause() {
        assertThrows(NullPointerException.class,
                () -> DirectGenerationFailures.wrap(null, "log message", "Direct generation failed", null));
    }
}
