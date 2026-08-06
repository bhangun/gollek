/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectPromptPreparationTest {

    @Test
    void preparesTextPromptWithTokenizerConfigAndProfile() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(new long[] { 4, 5, 6 });
        ModelConfig config = new ModelConfig();
        InferenceProfile profile = new InferenceProfile("test", false);

        DirectPromptPreparation preparation = DirectPromptPreparation.text(
                tokenizer, config, null, "hello", profile, DirectPromptPreparation.DebugOptions.none());

        assertSame(tokenizer, preparation.tokenizer());
        assertSame(config, preparation.config());
        assertArrayEquals(new long[] { 4, 5, 6 }, preparation.ids());
        assertTrue(profile.tokenizeNanos >= 0L);
        assertNotNull(tokenizer.lastOptions);
    }

    @Test
    void preparesPretokenizedPromptWithoutCopyingInput() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(new long[] { 1 });
        ModelConfig config = new ModelConfig();
        long[] inputIds = new long[] { 7, 8 };

        DirectPromptPreparation preparation = DirectPromptPreparation.pretokenized(
                tokenizer, config, inputIds, DirectPromptPreparation.DebugOptions.none());

        assertSame(tokenizer, preparation.tokenizer());
        assertSame(config, preparation.config());
        assertSame(inputIds, preparation.ids());
    }

    @Test
    void printsTextPromptDebugWhenEnabled() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(new long[] { 9 });

        String output = captureStdout(() -> DirectPromptPreparation.text(
                tokenizer,
                new ModelConfig(),
                null,
                "hello",
                null,
                DirectPromptPreparation.DebugOptions.text(true, "[DEBUG-S]", true)));

        assertTrue(output.contains("[DEBUG-S] 3: get tokenizer/config"));
        assertTrue(output.contains("[DEBUG-S] 5: tokenize"));
        assertTrue(output.contains("[DEBUG-PROMPT-TEXT] hello"));
        assertTrue(output.contains("[DEBUG-S] 6: tokens=1"));
    }

    @Test
    void printsPretokenizedDebugWhenEnabled() {
        String output = captureStdout(() -> DirectPromptPreparation.pretokenized(
                new CapturingTokenizer(new long[] { 1 }),
                new ModelConfig(),
                new long[] { 10, 11, 12 },
                DirectPromptPreparation.DebugOptions.pretokenized(true, "[DEBUG] pretokenized")));

        assertTrue(output.contains("[DEBUG] pretokenized tokens=3"));
    }

    private static String captureStdout(Runnable action) {
        PrintStream original = System.out;
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(out, true, StandardCharsets.UTF_8));
            action.run();
            return out.toString(StandardCharsets.UTF_8);
        } finally {
            System.setOut(original);
        }
    }

    private static final class CapturingTokenizer implements Tokenizer {
        private final long[] ids;
        private EncodeOptions lastOptions;

        private CapturingTokenizer(long[] ids) {
            this.ids = ids;
        }

        @Override
        public long[] encode(String text, EncodeOptions options) {
            this.lastOptions = options;
            return ids;
        }

        @Override
        public String decode(long[] tokens, DecodeOptions options) {
            return "";
        }

        @Override
        public int vocabSize() {
            return 0;
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
