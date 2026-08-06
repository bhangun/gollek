/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GenerationTokenDebugTest {
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
    void printableTextEscapesNewlinesAndNulls() {
        assertEquals("<null>", GenerationTokenDebug.printableText(null));
        assertEquals("a\\nb", GenerationTokenDebug.printableText("a\nb"));
    }

    @Test
    void tokenSequencePrintsDecodedTokens() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(Map.of(1, "hello", 2, "line\nbreak"));

        String output = captureOutput(() -> GenerationTokenDebug.tokenSequence(tokenizer, new long[] { 1, 2 },
                "prompt"));

        assertEquals("[DEBUG-TOKENS] prompt: [1:hello] [2:line\\nbreak]" + System.lineSeparator(), output);
    }

    @Test
    void chosenTokenHonorsVerboseProperty() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(Map.of(3, "chosen"));

        System.clearProperty("gollek.verbose");
        assertEquals("", captureOutput(() -> GenerationTokenDebug.chosenToken(tokenizer, 3, 1)));

        System.setProperty("gollek.verbose", "true");
        assertEquals("[DEBUG-CHOSEN] step=1 token=3 text=chosen" + System.lineSeparator(),
                captureOutput(() -> GenerationTokenDebug.chosenToken(tokenizer, 3, 1)));
    }

    private static String captureOutput(Runnable runnable) {
        PrintStream originalOut = System.out;
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            runnable.run();
            return output.toString(StandardCharsets.UTF_8);
        } finally {
            System.setOut(originalOut);
        }
    }

    private static final class CapturingTokenizer implements Tokenizer {
        private final Map<Integer, String> decodedTokens;

        private CapturingTokenizer(Map<Integer, String> decodedTokens) {
            this.decodedTokens = decodedTokens;
        }

        @Override
        public long[] encode(String text, EncodeOptions options) {
            return new long[0];
        }

        @Override
        public String decode(long[] tokens, DecodeOptions options) {
            if (tokens == null || tokens.length == 0) {
                return "";
            }
            return decodedTokens.getOrDefault((int) tokens[0], "token-" + tokens[0]);
        }

        @Override
        public int vocabSize() {
            return 16;
        }

        @Override
        public int bosTokenId() {
            return -1;
        }

        @Override
        public int eosTokenId() {
            return -1;
        }

        @Override
        public int padTokenId() {
            return -1;
        }

        @Override
        public int[] allStopTokenIds() {
            return new int[0];
        }
    }
}
