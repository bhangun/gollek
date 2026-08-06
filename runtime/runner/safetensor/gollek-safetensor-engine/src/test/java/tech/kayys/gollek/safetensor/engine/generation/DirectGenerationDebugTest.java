/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectGenerationDebugTest {
    private String previousVerbose;

    @BeforeEach
    void captureVerboseProperty() {
        previousVerbose = System.getProperty("gollek.verbose");
    }

    @AfterEach
    void restoreVerboseProperty() {
        if (previousVerbose == null) {
            System.clearProperty("gollek.verbose");
        } else {
            System.setProperty("gollek.verbose", previousVerbose);
        }
    }

    @Test
    void createsTextAndStreamDebugFromVerboseProperty() {
        System.setProperty("gollek.verbose", "true");

        DirectGenerationDebug text = DirectGenerationDebug.text();
        DirectGenerationDebug stream = DirectGenerationDebug.stream();

        assertTrue(text.enabled());
        assertEquals("[DEBUG]", text.prefix());
        assertTrue(stream.enabled());
        assertEquals("[DEBUG-S]", stream.prefix());
    }

    @Test
    void createsDisabledDebugWhenVerbosePropertyIsNotSet() {
        System.clearProperty("gollek.verbose");

        assertFalse(DirectGenerationDebug.text().enabled());
        assertFalse(DirectGenerationDebug.stream().enabled());
    }

    @Test
    void buildsPromptDebugOptions() {
        DirectGenerationDebug debug = DirectGenerationDebug.of(true, "[X]");

        DirectPromptPreparation.DebugOptions text = debug.textPrompt(true);
        DirectPromptPreparation.DebugOptions pretokenized = debug.pretokenized("[X] pretokenized");

        assertTrue(text.enabled());
        assertEquals("[X]", text.prefix());
        assertTrue(text.printPromptText());
        assertTrue(pretokenized.enabled());
        assertEquals("[X] pretokenized", pretokenized.pretokenizedLabel());
    }

    @Test
    void returnsChosenTokenObserverOnlyWhenEnabled() {
        assertNull(DirectGenerationDebug.of(false, "[X]").chosenTokenObserver(null));
        assertNotNull(DirectGenerationDebug.of(true, "[X]").chosenTokenObserver(null));
    }

    @Test
    void printsStepOnlyWhenEnabled() {
        PrintStream originalOut = System.out;
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));

            DirectGenerationDebug.of(false, "[X]").step(1, "hidden");
            DirectGenerationDebug.of(true, "[X]").step(2, "visible");

            assertEquals("[X] 2: visible" + System.lineSeparator(),
                    output.toString(StandardCharsets.UTF_8));
        } finally {
            System.setOut(originalOut);
        }
    }
}
