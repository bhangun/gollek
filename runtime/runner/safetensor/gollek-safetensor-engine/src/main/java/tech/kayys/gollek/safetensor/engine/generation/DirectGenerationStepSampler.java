/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.canUseDirectGreedySampling;
import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.initializePromptFrequencies;
import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.sampleGreedyFromTensor;
import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.sampleNextToken;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;

/**
 * Performs prefill/decode forward passes and converts logits into the next
 * token for one direct generation run.
 */
final class DirectGenerationStepSampler {
    private final Supplier<DirectForwardPass> forwardPass;
    private final Supplier<TokenSampler> tokenSampler;

    DirectGenerationStepSampler(Supplier<DirectForwardPass> forwardPass, Supplier<TokenSampler> tokenSampler) {
        this.forwardPass = forwardPass;
        this.tokenSampler = tokenSampler;
    }

    SamplingState createState(DirectLoadedModel model, GenerationConfig cfg, long[] promptTokenIds, Set<Integer> stops,
            SamplingMode mode) {
        ModelConfig config = model.config();
        DirectForwardPass resolvedForwardPass = forwardPass.get();
        boolean directGreedy = canUseDirectGreedySampling(cfg);
        int[] frequencies = directGreedy ? null : initializePromptFrequencies(config, promptTokenIds);
        Random rng = directGreedy ? null : new Random();
        TokenSampler resolvedTokenSampler = directGreedy ? null : tokenSampler.get();
        TokenSamplingMasks samplingMasks = mode == SamplingMode.TOKENIZER_AWARE
                ? GenerationTokenPolicy.tokenSamplingMasksFor(model.baseTokenSamplingMasks(), model.tokenizer(),
                        stops, config.getVocabSize(), cfg, model.runtimeTraits())
                : null;
        GenerationTokenValidationCache validationCache = mode == SamplingMode.TOKENIZER_AWARE
                ? new GenerationTokenValidationCache(config.getVocabSize())
                : null;
        return new SamplingState(mode, directGreedy, resolvedForwardPass, frequencies, rng, resolvedTokenSampler, stops,
                samplingMasks, validationCache);
    }

    StepResult prefill(long[] inputIds, DirectLoadedModel model, KVCacheManager.KVCacheSession session,
            GenerationConfig cfg, SamplingState sampling) {
        ModelConfig config = model.config();
        ModelArchitecture arch = model.architecture();
        DirectForwardPass resolvedForwardPass = forwardPass(sampling);
        long tForward0 = System.nanoTime();
        if (sampling.directGreedy()) {
            AccelTensor logits = resolvedForwardPass.prefillLogitsTensor(inputIds, model.weights(), config, arch,
                    session);
            long forwardNanos = System.nanoTime() - tForward0;
            long tSample0 = System.nanoTime();
            int next = sampleGreedy(logits, model, sampling, true);
            return new StepResult(next, forwardNanos, System.nanoTime() - tSample0);
        }

        float[] logits = resolvedForwardPass.prefill(inputIds, model.weights(), config, arch, session);
        long forwardNanos = System.nanoTime() - tForward0;
        long tSample0 = System.nanoTime();
        int next = sampleLogits(logits, model, cfg, sampling, true);
        return new StepResult(next, forwardNanos, System.nanoTime() - tSample0);
    }

    StepResult decode(int previousToken, int decodeStartPos, DirectLoadedModel model,
            KVCacheManager.KVCacheSession session, GenerationConfig cfg, SamplingState sampling) {
        ModelConfig config = model.config();
        ModelArchitecture arch = model.architecture();
        DirectForwardPass resolvedForwardPass = forwardPass(sampling);
        long tForward0 = System.nanoTime();
        if (sampling.directGreedy()) {
            AccelTensor logits = resolvedForwardPass.decodeLogitsTensor(previousToken, decodeStartPos, model.weights(),
                    config, arch, session, true);
            long forwardNanos = System.nanoTime() - tForward0;
            long tSample0 = System.nanoTime();
            int next = sampleGreedy(logits, model, sampling, false);
            return new StepResult(next, forwardNanos, System.nanoTime() - tSample0);
        }

        float[] logits = resolvedForwardPass.decode(previousToken, decodeStartPos, model.weights(), config, arch,
                session);
        long forwardNanos = System.nanoTime() - tForward0;
        long tSample0 = System.nanoTime();
        int next = sampleLogits(logits, model, cfg, sampling, false);
        return new StepResult(next, forwardNanos, System.nanoTime() - tSample0);
    }

    private int sampleGreedy(AccelTensor logits, DirectLoadedModel model, SamplingState sampling, boolean firstStep) {
        if (sampling.mode() == SamplingMode.TOKENIZER_AWARE) {
            return sampleGreedyFromTensor(logits, model.config(), model.tokenizer(), model.runtimeTraits(), firstStep,
                    sampling.stops(), sampling.samplingMasks(), sampling.validationCache());
        }
        return sampleGreedyFromTensor(logits, model.config());
    }

    private int sampleLogits(float[] logits, DirectLoadedModel model, GenerationConfig cfg, SamplingState sampling,
            boolean firstStep) {
        TokenSampler resolvedTokenSampler = tokenSampler(sampling);
        if (sampling.mode() == SamplingMode.TOKENIZER_AWARE) {
            Tokenizer tokenizer = model.tokenizer();
            return sampleNextToken(logits, tokenizer, cfg, model.config(), model.runtimeTraits(),
                    sampling.frequencies(), sampling.rng(), firstStep, sampling.stops(), resolvedTokenSampler,
                    sampling.validationCache(), sampling.samplingMasks());
        }
        return resolvedTokenSampler.sample(logits, cfg, model.config(), sampling.frequencies(), sampling.rng());
    }

    private TokenSampler tokenSampler(SamplingState sampling) {
        TokenSampler resolved = sampling.tokenSampler();
        return resolved != null ? resolved : tokenSampler.get();
    }

    private DirectForwardPass forwardPass(SamplingState sampling) {
        DirectForwardPass resolved = sampling.forwardPass();
        return resolved != null ? resolved : forwardPass.get();
    }

    enum SamplingMode {
        RAW_PRETOKENIZED,
        TOKENIZER_AWARE
    }

    record SamplingState(
            SamplingMode mode,
            boolean directGreedy,
            DirectForwardPass forwardPass,
            int[] frequencies,
            Random rng,
            TokenSampler tokenSampler,
            Set<Integer> stops,
            TokenSamplingMasks samplingMasks,
            GenerationTokenValidationCache validationCache) {
    }

    record StepResult(int token, long forwardNanos, long samplingNanos) {
    }
}
