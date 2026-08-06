/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectLoadedModelAcquirerTest {

    @Test
    void returnsAlreadyLoadedModelWithoutLoading() {
        AtomicBoolean loadCalled = new AtomicBoolean(false);
        AtomicLong recordedNanos = new AtomicLong(-1L);

        String model = DirectLoadedModelAcquirer.require(
                () -> "loaded",
                () -> loadCalled.set(true),
                recordedNanos::set,
                false,
                "[DEBUG]");

        assertEquals("loaded", model);
        assertFalse(loadCalled.get());
        assertEquals(-1L, recordedNanos.get());
    }

    @Test
    void loadsMissingModelAndRecordsLoadTime() {
        AtomicReference<String> modelRef = new AtomicReference<>();
        AtomicLong recordedNanos = new AtomicLong(-1L);

        String model = DirectLoadedModelAcquirer.require(
                modelRef::get,
                () -> modelRef.set("loaded"),
                recordedNanos::set,
                false,
                "[DEBUG]");

        assertEquals("loaded", model);
        assertTrue(recordedNanos.get() >= 0L);
    }

    @Test
    void failsWhenLoadDoesNotPopulateModel() {
        assertThrows(RuntimeException.class,
                () -> DirectLoadedModelAcquirer.require(
                        () -> null,
                        () -> {
                        },
                        null,
                        false,
                        "[DEBUG]"));
    }
}
