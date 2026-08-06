/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.util.BitSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GenerationTokenPolicyTest {

    @Test
    void greedyTensorSamplingUsesCheapRetryAfterMaskedWinner() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        GenerationGreedyArgmax.setNativeArgmaxDisabledForTest(true);
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");
            BitSet continuationMask = new BitSet();
            continuationMask.set(2);
            TokenSamplingMasks masks = new TokenSamplingMasks(new BitSet(), continuationMask);
            AccelTensor logits = AccelTensor.fromFloatArray(new float[] { 0.25f, 2.0f, 5.0f, 3.0f }, 4);

            int selected = GenerationTokenPolicy.sampleGreedyFromTensor(
                    logits,
                    new ModelConfig(),
                    null,
                    ModelRuntimeTraits.EMPTY,
                    false,
                    Set.of(),
                    masks);

            assertEquals(3, selected);
            String summary = profile.summary("cpu");
            assertTrue(summary.contains("argmax_paths={java_memory_segment=2}"));
            Map<String, Object> metadata = profile.metadata("cpu");
            assertEquals(2, metadata.get("profile_argmax_path_java_memory_segment_count"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProperty("gollek.profile", previousProfile);
            GenerationGreedyArgmax.setNativeArgmaxDisabledForTest(null);
        }
    }

    @Test
    void greedyTensorSamplingFallsBackToFullMaskAfterCheapRetryCapacity() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        GenerationGreedyArgmax.setNativeArgmaxDisabledForTest(true);
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");
            BitSet continuationMask = new BitSet();
            for (int tokenId = 0; tokenId < 8; tokenId++) {
                continuationMask.set(tokenId);
            }
            TokenSamplingMasks masks = new TokenSamplingMasks(new BitSet(), continuationMask);
            AccelTensor logits = AccelTensor.fromFloatArray(
                    new float[] { 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f }, 10);

            int selected = GenerationTokenPolicy.sampleGreedyFromTensor(
                    logits,
                    new ModelConfig(),
                    null,
                    ModelRuntimeTraits.EMPTY,
                    false,
                    Set.of(),
                    masks);

            assertEquals(8, selected);
            String summary = profile.summary("cpu");
            assertTrue(summary.contains("argmax_paths={java_memory_segment=8, java_memory_segment_mask=1}"));
            Map<String, Object> metadata = profile.metadata("cpu");
            assertEquals(8, metadata.get("profile_argmax_path_java_memory_segment_count"));
            assertEquals(1, metadata.get("profile_argmax_path_java_memory_segment_mask_count"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProperty("gollek.profile", previousProfile);
            GenerationGreedyArgmax.setNativeArgmaxDisabledForTest(null);
        }
    }

    @Test
    void greedyTensorSamplingKeepsFullMaskFallbackAfterValidationReject() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        GenerationGreedyArgmax.setNativeArgmaxDisabledForTest(true);
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");
            BitSet firstStepMask = new BitSet();
            for (int tokenId = 0; tokenId < 8; tokenId++) {
                firstStepMask.set(tokenId);
            }
            TokenSamplingMasks masks = new TokenSamplingMasks(firstStepMask, new BitSet());
            CountingTokenizer tokenizer = new CountingTokenizer(Map.of(), Map.of(8, "<bad>"), 10);
            AccelTensor logits = AccelTensor.fromFloatArray(
                    new float[] { 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f }, 10);

            int selected = GenerationTokenPolicy.sampleGreedyFromTensor(
                    logits,
                    new ModelConfig(),
                    tokenizer,
                    ModelRuntimeTraits.EMPTY,
                    true,
                    Set.of(),
                    masks);

            assertEquals(9, selected);
            assertEquals(2, tokenizer.decodeCalls());
            String summary = profile.summary("cpu");
            assertTrue(summary.contains("argmax_paths={java_memory_segment=8, java_memory_segment_mask=2}"));
            Map<String, Object> metadata = profile.metadata("cpu");
            assertEquals(8, metadata.get("profile_argmax_path_java_memory_segment_count"));
            assertEquals(2, metadata.get("profile_argmax_path_java_memory_segment_mask_count"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProperty("gollek.profile", previousProfile);
            GenerationGreedyArgmax.setNativeArgmaxDisabledForTest(null);
        }
    }

    @Test
    void greedyTensorSamplingReusesValidationCacheForRepeatedCandidate() {
        GenerationGreedyArgmax.setNativeArgmaxDisabledForTest(true);
        try {
            CountingTokenizer tokenizer = new CountingTokenizer();
            GenerationTokenValidationCache validationCache = new GenerationTokenValidationCache(4);
            ModelRuntimeTraits traits = ModelRuntimeTraits.builder()
                    .validateContinuationTokensByDecode()
                    .rejectEmptyDecodedTokens()
                    .build();

            assertEquals(2, sampleRepeatedCandidate(tokenizer, traits, validationCache));
            assertEquals(2, sampleRepeatedCandidate(tokenizer, traits, validationCache));
            assertEquals(1, tokenizer.decodeCalls());
        } finally {
            GenerationGreedyArgmax.setNativeArgmaxDisabledForTest(null);
        }
    }

    @Test
    void validationCacheReusesSpecialTokenLookupAcrossDifferentCandidates() {
        CountingTokenizer tokenizer = new CountingTokenizer(Map.of("<bad-a>", 2, "<bad-b>", 3));
        GenerationTokenValidationCache validationCache = new GenerationTokenValidationCache(4);

        assertTrue(validationCache.shouldRejectSampledToken(
                2, tokenizer, ModelRuntimeTraits.EMPTY, false, Set.of()));
        assertTrue(validationCache.shouldRejectSampledToken(
                3, tokenizer, ModelRuntimeTraits.EMPTY, false, Set.of()));
        assertEquals(1, tokenizer.specialTokenCalls());
    }

    @Test
    void sampledLogitPathUsesPrebuiltContinuationMask() {
        float[] logits = new float[] { 0.25f, 1.0f, 9.0f, 3.0f };
        BitSet continuationMask = new BitSet();
        continuationMask.set(2);
        TokenSamplingMasks masks = new TokenSamplingMasks(new BitSet(), continuationMask);
        RecordingTokenSampler tokenSampler = new RecordingTokenSampler(3);

        int selected = GenerationTokenPolicy.sampleNextToken(
                logits,
                null,
                GenerationConfig.defaults(),
                new ModelConfig(),
                ModelRuntimeTraits.EMPTY,
                null,
                new Random(1L),
                false,
                Set.of(),
                tokenSampler,
                null,
                masks);

        assertEquals(3, selected);
        assertEquals(Float.NEGATIVE_INFINITY, tokenSampler.observedLogit(2));
    }

    private static int sampleRepeatedCandidate(Tokenizer tokenizer, ModelRuntimeTraits traits,
            GenerationTokenValidationCache validationCache) {
        AccelTensor logits = AccelTensor.fromFloatArray(new float[] { 0.25f, 1.0f, 4.0f, 3.0f }, 4);
        return GenerationTokenPolicy.sampleGreedyFromTensor(
                logits,
                new ModelConfig(),
                tokenizer,
                traits,
                false,
                Set.of(),
                null,
                validationCache);
    }

    private static void restoreProperty(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }

    private static final class RecordingTokenSampler extends TokenSampler {
        private final int selectedToken;
        private float[] observedLogits = new float[0];

        private RecordingTokenSampler(int selectedToken) {
            this.selectedToken = selectedToken;
        }

        @Override
        public int sample(float[] logits, GenerationConfig config, ModelConfig modelConfig, int[] freq, Random rng) {
            observedLogits = logits.clone();
            return selectedToken;
        }

        float observedLogit(int tokenId) {
            return observedLogits[tokenId];
        }
    }

    private static final class CountingTokenizer implements Tokenizer {
        private final AtomicInteger decodeCalls = new AtomicInteger();
        private final AtomicInteger specialTokenCalls = new AtomicInteger();
        private final Map<String, Integer> specialTokens;
        private final Map<Integer, String> decodedTokens;
        private final int vocabSize;

        private CountingTokenizer() {
            this(Map.of());
        }

        private CountingTokenizer(Map<String, Integer> specialTokens) {
            this(specialTokens, Map.of(), 4);
        }

        private CountingTokenizer(Map<String, Integer> specialTokens, Map<Integer, String> decodedTokens,
                int vocabSize) {
            this.specialTokens = specialTokens;
            this.decodedTokens = decodedTokens;
            this.vocabSize = vocabSize;
        }

        @Override
        public long[] encode(String text, EncodeOptions options) {
            return new long[0];
        }

        @Override
        public String decode(long[] tokens, DecodeOptions options) {
            decodeCalls.incrementAndGet();
            return decodedTokens.getOrDefault((int) tokens[0], "token-" + tokens[0]);
        }

        @Override
        public int vocabSize() {
            return vocabSize;
        }

        @Override
        public int bosTokenId() {
            return 0;
        }

        @Override
        public int eosTokenId() {
            return 3;
        }

        @Override
        public int padTokenId() {
            return 1;
        }

        @Override
        public int[] allStopTokenIds() {
            return new int[] { 3 };
        }

        @Override
        public Map<String, Integer> specialTokens() {
            specialTokenCalls.incrementAndGet();
            return specialTokens;
        }

        int decodeCalls() {
            return decodeCalls.get();
        }

        int specialTokenCalls() {
            return specialTokenCalls.get();
        }
    }
}
