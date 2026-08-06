/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.util.BitSet;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GenerationTokenValidationPolicyTest {

    @Test
    void masksBosPadSpecialsAndOnlyFirstStepStops() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(
                Map.of("<bad>", 3, "<allowed>", 4, "<stop-special>", 5),
                Map.of());
        ModelRuntimeTraits traits = traits(false, Set.of("<allowed>"), false, false);

        BitSet firstStep = GenerationTokenValidationPolicy
                .buildDisallowedTokenMask(tokenizer, true, Set.of(5, 6), 10, traits);
        BitSet continuation = GenerationTokenValidationPolicy
                .buildDisallowedTokenMask(tokenizer, false, Set.of(5, 6), 10, traits);

        assertTrue(firstStep.get(0));
        assertTrue(firstStep.get(1));
        assertTrue(firstStep.get(3));
        assertFalse(firstStep.get(4));
        assertTrue(firstStep.get(5));
        assertTrue(firstStep.get(6));
        assertFalse(continuation.get(5));
        assertFalse(continuation.get(6));
    }

    @Test
    void disallowedContinuationTokenRespectsAllowedControlsAndStops() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(
                Map.of("<bad>", 3, "<allowed>", 4),
                Map.of());
        ModelRuntimeTraits traits = traits(false, Set.of("<allowed>"), false, false);

        assertTrue(GenerationTokenValidationPolicy
                .isDisallowedContinuationToken(0, tokenizer, false, Set.of(), traits));
        assertTrue(GenerationTokenValidationPolicy
                .isDisallowedContinuationToken(3, tokenizer, false, Set.of(), traits));
        assertFalse(GenerationTokenValidationPolicy
                .isDisallowedContinuationToken(4, tokenizer, false, Set.of(), traits));
        assertTrue(GenerationTokenValidationPolicy
                .isDisallowedContinuationToken(5, tokenizer, true, Set.of(5), traits));
        assertFalse(GenerationTokenValidationPolicy
                .isDisallowedContinuationToken(5, tokenizer, false, Set.of(5), traits));
    }

    @Test
    void rejectsGemma4AssistantControlOnFirstStep() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(Map.of(), Map.of(7, "assistant"));
        ModelRuntimeTraits traits = traits(true, Set.of(), true, true);

        assertTrue(GenerationTokenValidationPolicy
                .shouldRejectSampledToken(7, tokenizer, traits, true, Set.of()));
    }

    @Test
    void rejectsEmptyDecodedTokenButKeepsWhitespaceToken() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(Map.of(), Map.of(8, "", 9, " "));
        ModelRuntimeTraits traits = traits(false, Set.of(), true, true);

        assertTrue(GenerationTokenValidationPolicy
                .shouldRejectSampledToken(8, tokenizer, traits, false, Set.of()));
        assertFalse(GenerationTokenValidationPolicy
                .shouldRejectSampledToken(9, tokenizer, traits, true, Set.of()));
    }

    @Test
    void continuationDecodeValidationCanBeSkipped() {
        CapturingTokenizer tokenizer = new CapturingTokenizer(Map.of(), Map.of(10, "<unused1>"));
        ModelRuntimeTraits traits = traits(false, Set.of(), false, true);

        assertFalse(GenerationTokenValidationPolicy
                .shouldRejectSampledToken(10, tokenizer, traits, false, Set.of()));
    }

    private static ModelRuntimeTraits traits(
            boolean gemma4,
            Set<String> allowedControls,
            boolean validateContinuation,
            boolean rejectEmpty) {
        return ModelRuntimeTraits.builder()
                .nativeBf16Matvec(gemma4)
                .allowedControlTokenTexts(allowedControls)
                .validateContinuationTokensByDecode(validateContinuation)
                .rejectEmptyDecodedTokens(rejectEmpty)
                .build();
    }

    private static final class CapturingTokenizer implements Tokenizer {
        private final Map<String, Integer> specialTokens;
        private final Map<Integer, String> decodedTokens;

        private CapturingTokenizer(Map<String, Integer> specialTokens, Map<Integer, String> decodedTokens) {
            this.specialTokens = specialTokens;
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
            return 1;
        }

        @Override
        public int eosTokenId() {
            return 2;
        }

        @Override
        public int padTokenId() {
            return 0;
        }

        @Override
        public int[] allStopTokenIds() {
            return new int[] { 2 };
        }

        @Override
        public Map<String, Integer> specialTokens() {
            return specialTokens;
        }
    }
}
