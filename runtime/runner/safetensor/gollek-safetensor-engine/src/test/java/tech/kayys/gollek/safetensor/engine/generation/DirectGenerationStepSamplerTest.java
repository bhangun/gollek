/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.nio.file.Path;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression coverage for direct generation sampling-state setup.
 */
class DirectGenerationStepSamplerTest {

    @Test
    void resolvesTokenSamplerOnceForNonGreedyState() {
        GenerationConfig cfg = GenerationConfig.builder()
                .strategy(GenerationConfig.SamplingStrategy.TOP_K)
                .build();
        DirectLoadedModel model = model();
        CountingTokenSampler tokenSampler = new CountingTokenSampler();
        AtomicInteger tokenSamplerLookups = new AtomicInteger();
        DeterministicForwardPass forwardPass = new DeterministicForwardPass();
        AtomicInteger forwardPassLookups = new AtomicInteger();
        DirectGenerationStepSampler stepSampler = new DirectGenerationStepSampler(
                () -> {
                    forwardPassLookups.incrementAndGet();
                    return forwardPass;
                },
                () -> {
                    tokenSamplerLookups.incrementAndGet();
                    return tokenSampler;
                });

        DirectGenerationStepSampler.SamplingState sampling = stepSampler.createState(
                model, cfg, new long[] { 1L }, Set.of(), DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED);
        DirectGenerationStepSampler.StepResult prefill =
                stepSampler.prefill(new long[] { 1L }, model, null, cfg, sampling);
        DirectGenerationStepSampler.StepResult decode =
                stepSampler.decode(prefill.token(), 1, model, null, cfg, sampling);

        assertEquals(1, prefill.token());
        assertEquals(2, decode.token());
        assertEquals(1, forwardPassLookups.get());
        assertEquals(1, forwardPass.prefillCalls());
        assertEquals(1, forwardPass.decodeCalls());
        assertEquals(1, tokenSamplerLookups.get());
        assertEquals(2, tokenSampler.calls());
    }

    @Test
    void skipsTokenSamplerResolutionForDirectGreedyState() {
        DirectLoadedModel model = model();
        AtomicInteger tokenSamplerLookups = new AtomicInteger();
        DirectGenerationStepSampler stepSampler = new DirectGenerationStepSampler(
                DeterministicForwardPass::new,
                () -> {
                    tokenSamplerLookups.incrementAndGet();
                    return new CountingTokenSampler();
                });

        DirectGenerationStepSampler.SamplingState sampling = stepSampler.createState(
                model, GenerationConfig.defaults(), new long[] { 1L }, Set.of(),
                DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED);

        assertEquals(0, tokenSamplerLookups.get());
        assertTrue(sampling.directGreedy());
    }

    private static DirectLoadedModel model() {
        return new DirectLoadedModel(
                Path.of("test-model"),
                Map.of(),
                new EmptyTokenizer(),
                "test-model",
                false,
                null,
                "none",
                null,
                new ModelConfig(),
                new TestModelArchitecture(),
                ModelRuntimeTraits.EMPTY,
                null);
    }

    /**
     * Forward-pass stub that returns small logits arrays for non-greedy sampler
     * tests.
     */
    private static final class DeterministicForwardPass extends DirectForwardPass {
        private int prefillCalls;
        private int decodeCalls;

        @Override
        public float[] prefill(long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config,
                ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
            prefillCalls++;
            return new float[] { 0.0f, 2.0f, 1.0f };
        }

        @Override
        public float[] decode(long tokenId, int startPos, Map<String, AccelTensor> weights, ModelConfig config,
                ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
            decodeCalls++;
            return new float[] { 0.0f, 1.0f, 2.0f };
        }

        private int prefillCalls() {
            return prefillCalls;
        }

        private int decodeCalls() {
            return decodeCalls;
        }
    }

    /**
     * Token sampler stub that picks the current argmax and counts calls.
     */
    private static final class CountingTokenSampler extends TokenSampler {
        private int calls;

        @Override
        public int sample(float[] logits, GenerationConfig config, ModelConfig modelConfig, int[] freq, Random rng) {
            calls++;
            int best = 0;
            for (int i = 1; i < logits.length; i++) {
                if (logits[i] > logits[best]) {
                    best = i;
                }
            }
            return best;
        }

        private int calls() {
            return calls;
        }
    }

    /**
     * Tokenizer fixture for model construction when tests do not decode text.
     */
    private static final class EmptyTokenizer implements Tokenizer {
        @Override
        public long[] encode(String text, EncodeOptions options) {
            return new long[0];
        }

        @Override
        public String decode(long[] tokens, DecodeOptions options) {
            return "";
        }

        @Override
        public int vocabSize() {
            return 3;
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
