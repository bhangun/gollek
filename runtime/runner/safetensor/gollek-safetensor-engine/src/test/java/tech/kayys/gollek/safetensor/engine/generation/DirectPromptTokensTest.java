/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectPromptTokensTest {

    @Test
    void encodesWithRuntimeTraitBosPolicy() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(new long[] { 1, 2, 3 });
        ModelRuntimeTraits traits = ModelRuntimeTraits.builder()
                .promptBosPolicy(ModelRuntimeTraits.PromptBosPolicy.TURN_AWARE)
                .turnPromptPrefixes(Set.of("<|turn>"))
                .build();

        DirectPromptTokens tokens = DirectPromptTokens.encode(
                tokenizer, new ModelConfig(), traits, "hello", new InferenceProfile("test", false));

        assertArrayEquals(new long[] { 1, 2, 3 }, tokens.ids());
        assertTrue(tokenizer.lastOptions.addBos);
    }

    @Test
    void skipsBosWhenPromptAlreadyLooksLikeGemmaTurnPrompt() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(new long[] { 4 });
        ModelRuntimeTraits traits = ModelRuntimeTraits.builder()
                .promptBosPolicy(ModelRuntimeTraits.PromptBosPolicy.TURN_AWARE)
                .turnPromptPrefixes(Set.of("<|turn>"))
                .build();

        DirectPromptTokens.encode(tokenizer, new ModelConfig(), traits, "<|turn>user\nhello", null);

        assertFalse(tokenizer.lastOptions.addBos);
    }

    @Test
    void rejectsEmptyEncodedPrompt() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(new long[0]);

        assertThrows(IllegalArgumentException.class,
                () -> DirectPromptTokens.encode(tokenizer, new ModelConfig(), null, "", null));
    }

    @Test
    void rejectsNullPretokenizedPrompt() {
        assertThrows(IllegalArgumentException.class, () -> DirectPromptTokens.of(null));
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
