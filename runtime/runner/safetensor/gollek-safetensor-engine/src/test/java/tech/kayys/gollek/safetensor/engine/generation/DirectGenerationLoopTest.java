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
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Regression coverage for the direct generation loop's per-token hot path.
 */
class DirectGenerationLoopTest {

    @Test
    void reusesStepSamplerWithinSingleRun() {
        GenerationConfig cfg = GenerationConfig.builder().maxNewTokens(4).build();
        DirectLoadedModel model = model();
        DeterministicForwardPass forwardPass = new DeterministicForwardPass();
        DirectGenerationStepSampler sampler =
                new DirectGenerationStepSampler(() -> forwardPass, TokenSampler::new);
        DirectGenerationStepSampler.SamplingState sampling = sampler.createState(
                model, cfg, new long[] { 0L }, Set.of(), DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED);
        AtomicInteger samplerLookups = new AtomicInteger();
        DirectGenerationLoop loop = new DirectGenerationLoop(() -> {
            samplerLookups.incrementAndGet();
            return sampler;
        });

        DirectGenerationLoop.Result result = loop.run(new DirectGenerationLoop.Request(
                model,
                cfg,
                null,
                sampling,
                Set.of(),
                1,
                1,
                System.nanoTime(),
                null,
                true,
                true,
                true,
                null,
                null,
                null));

        assertEquals("Jakarta!", result.text());
        assertEquals(List.of(1L, 2L, 3L, 4L), result.generatedTokenIds());
        assertEquals(4, result.completionTokens());
        assertEquals(3, result.decodeSteps());
        assertEquals(3, forwardPass.decodeCalls());
        assertEquals(1, samplerLookups.get());
    }

    @Test
    void skipsStepSamplerLookupWhenFirstTokenCompletesRun() {
        GenerationConfig cfg = GenerationConfig.builder().maxNewTokens(1).build();
        DirectLoadedModel model = model();
        DirectGenerationStepSampler.SamplingState sampling = sampling(model, cfg);
        AtomicInteger samplerLookups = new AtomicInteger();
        DirectGenerationLoop loop = new DirectGenerationLoop(() -> {
            samplerLookups.incrementAndGet();
            return new DirectGenerationStepSampler(DeterministicForwardPass::new, TokenSampler::new);
        });

        DirectGenerationLoop.Result result = loop.run(new DirectGenerationLoop.Request(
                model,
                cfg,
                null,
                sampling,
                Set.of(),
                1,
                1,
                System.nanoTime(),
                null,
                true,
                true,
                true,
                null,
                null,
                null));

        assertEquals("Jak", result.text());
        assertEquals(List.of(1L), result.generatedTokenIds());
        assertEquals(1, result.completionTokens());
        assertEquals(0, result.decodeSteps());
        assertEquals(0, samplerLookups.get());
    }

    @Test
    void stopsOnConfiguredStopString() {
        GenerationConfig cfg = GenerationConfig.builder()
                .maxNewTokens(4)
                .stopStrings(List.of("Jakarta"))
                .build();
        DirectLoadedModel model = model();
        DeterministicForwardPass forwardPass = new DeterministicForwardPass();
        DirectGenerationStepSampler sampler =
                new DirectGenerationStepSampler(() -> forwardPass, TokenSampler::new);
        DirectGenerationStepSampler.SamplingState sampling = sampler.createState(
                model, cfg, new long[] { 0L }, Set.of(), DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED);
        DirectGenerationLoop loop = new DirectGenerationLoop(() -> sampler);

        DirectGenerationLoop.Result result = loop.run(new DirectGenerationLoop.Request(
                model,
                cfg,
                null,
                sampling,
                Set.of(),
                1,
                1,
                System.nanoTime(),
                null,
                false,
                false,
                true,
                null,
                null,
                null));

        assertEquals("Jakarta", result.text());
        assertEquals(2, result.decodeSteps());
    }

    private static DirectLoadedModel model() {
        return new DirectLoadedModel(
                Path.of("test-model"),
                Map.of(),
                new TokenTextTokenizer(Map.of(
                        1L, "Jak",
                        2L, "ar",
                        3L, "ta",
                        4L, "!")),
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

    private static DirectGenerationStepSampler.SamplingState sampling(DirectLoadedModel model, GenerationConfig cfg) {
        return new DirectGenerationStepSampler(DeterministicForwardPass::new, TokenSampler::new)
                .createState(model, cfg, new long[] { 0L }, Set.of(),
                        DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED);
    }

    /**
     * Forward-pass stub that turns each previous token into deterministic logits
     * for the next token.
     */
    private static final class DeterministicForwardPass extends DirectForwardPass {
        private int decodeCalls;

        @Override
        public AccelTensor decodeLogitsTensor(long tokenId, int startPos, Map<String, AccelTensor> weights,
                ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache,
                boolean reuseLogitsOutput) {
            decodeCalls++;
            return logitsFor(nextToken(tokenId));
        }

        private int decodeCalls() {
            return decodeCalls;
        }

        private static int nextToken(long previousToken) {
            if (previousToken == 1L) {
                return 2;
            }
            if (previousToken == 2L) {
                return 3;
            }
            return 4;
        }

        private static AccelTensor logitsFor(int tokenId) {
            float[] logits = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            logits[tokenId] = 10.0f;
            return AccelTensor.fromFloatArray(logits, logits.length);
        }
    }

    /**
     * Tokenizer stub that decodes test token IDs into fixed text fragments.
     */
    private static final class TokenTextTokenizer implements Tokenizer {
        private final Map<Long, String> tokenText;

        private TokenTextTokenizer(Map<Long, String> tokenText) {
            this.tokenText = tokenText;
        }

        @Override
        public long[] encode(String text, EncodeOptions options) {
            return new long[0];
        }

        @Override
        public String decode(long[] tokens, DecodeOptions options) {
            StringBuilder text = new StringBuilder();
            for (long token : tokens) {
                text.append(tokenText.getOrDefault(token, ""));
            }
            return text.toString();
        }

        @Override
        public int vocabSize() {
            return 5;
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
