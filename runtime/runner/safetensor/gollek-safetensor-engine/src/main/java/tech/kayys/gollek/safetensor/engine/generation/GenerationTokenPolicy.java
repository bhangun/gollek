/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.runtime.ModelRuntimeTraitsResolver;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.lang.foreign.MemorySegment;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

final class GenerationTokenPolicy {
    private static final int GREEDY_RETRY_LIMIT = 8;
    private static final int GREEDY_REJECTED_CANDIDATE_CAPACITY = 8;
    private static final int GREEDY_FULL_MASK_RETRY_LIMIT = 8;

    private GenerationTokenPolicy() {
    }

    static boolean canUseDirectGreedySampling(GenerationConfig cfg) {
        return cfg.isPenaltyFreeGreedy();
    }

    static Set<Integer> buildBaseStopTokenIds(Tokenizer tokenizer, ModelConfig config) {
        Set<Integer> stops = new HashSet<>();
        if (tokenizer != null) {
            for (int id : tokenizer.allStopTokenIds()) {
                stops.add(id);
            }
        }
        if (config != null) {
            stops.addAll(config.eosTokenIds());
        }
        return Collections.unmodifiableSet(stops);
    }

    static Set<Integer> mergeStopTokenIds(Set<Integer> baseStops, GenerationConfig cfg) {
        List<Integer> requestStops = cfg.stopTokenIds();
        if (requestStops == null || requestStops.isEmpty()) {
            return baseStops;
        }
        Set<Integer> stops = new HashSet<>(baseStops.size() + requestStops.size());
        stops.addAll(baseStops);
        stops.addAll(requestStops);
        return stops;
    }

    static TokenSamplingMasks tokenSamplingMasksFor(TokenSamplingMasks baseMasks, Tokenizer tokenizer,
            Set<Integer> stops, int vocabSize, GenerationConfig cfg, ModelRuntimeTraits traits) {
        List<Integer> requestStops = cfg.stopTokenIds();
        if (requestStops == null || requestStops.isEmpty()) {
            return baseMasks;
        }
        return buildTokenSamplingMasks(tokenizer, stops, vocabSize, traits);
    }

    static TokenSamplingMasks buildTokenSamplingMasks(Tokenizer tokenizer, Set<Integer> stops, int vocabSize) {
        return GenerationTokenValidationPolicy.buildTokenSamplingMasks(tokenizer, stops, vocabSize, null);
    }

    static TokenSamplingMasks buildTokenSamplingMasks(Tokenizer tokenizer, Set<Integer> stops, int vocabSize,
            ModelRuntimeTraits traits) {
        return GenerationTokenValidationPolicy.buildTokenSamplingMasks(tokenizer, stops, vocabSize, traits);
    }

    static int[] initializePromptFrequencies(ModelConfig config, long[] tokenIds) {
        int[] freq = new int[config.vocabSize()];
        if (tokenIds == null || tokenIds.length == 0) {
            return freq;
        }
        for (long tokenId : tokenIds) {
            if (tokenId >= 0 && tokenId < freq.length) {
                freq[(int) tokenId]++;
            }
        }
        return freq;
    }

    static void recordSampleFrequency(int[] freq, int tokenId) {
        if (freq != null && tokenId >= 0 && tokenId < freq.length) {
            freq[tokenId]++;
        }
    }

    static int sampleNextToken(float[] logits, Tokenizer tokenizer, GenerationConfig cfg, ModelConfig config,
            ModelRuntimeTraits traits, int[] freq, Random rng, boolean firstStep, Set<Integer> stops,
            TokenSampler tokenSampler) {
        return sampleNextToken(logits, tokenizer, cfg, config, traits, freq, rng, firstStep, stops, tokenSampler, null);
    }

    static int sampleNextToken(float[] logits, Tokenizer tokenizer, GenerationConfig cfg, ModelConfig config,
            ModelRuntimeTraits traits, int[] freq, Random rng, boolean firstStep, Set<Integer> stops,
            TokenSampler tokenSampler, GenerationTokenValidationCache validationCache) {
        return sampleNextToken(logits, tokenizer, cfg, config, traits, freq, rng, firstStep, stops, tokenSampler,
                validationCache, null);
    }

    static int sampleNextToken(float[] logits, Tokenizer tokenizer, GenerationConfig cfg, ModelConfig config,
            ModelRuntimeTraits traits, int[] freq, Random rng, boolean firstStep, Set<Integer> stops,
            TokenSampler tokenSampler, GenerationTokenValidationCache validationCache, TokenSamplingMasks masks) {
        ModelRuntimeTraits effectiveTraits = effectiveTraits(config, traits);
        maskDisallowedContinuationTokens(logits, tokenizer, firstStep, stops, effectiveTraits, masks);
        for (int attempt = 0; attempt < 8; attempt++) {
            int next = tokenSampler.sample(logits, cfg, config, freq, rng);
            if (!shouldRejectSampledToken(next, tokenizer, effectiveTraits, firstStep, stops, validationCache)) {
                GenerationTokenDebug.sampleChoice("accept", next, tokenizer, firstStep, attempt);
                return next;
            }
            GenerationTokenDebug.sampleChoice("reject", next, tokenizer, firstStep, attempt);
            if (next >= 0 && next < logits.length) {
                logits[next] = Float.NEGATIVE_INFINITY;
            }
        }
        int fallback = tokenSampler.sample(logits, cfg, config, freq, rng);
        GenerationTokenDebug.sampleChoice("fallback", fallback, tokenizer, firstStep, 8);
        return fallback;
    }

    static int sampleGreedyFromTensor(AccelTensor logits, ModelConfig config) {
        try {
            return GenerationGreedyArgmax.select(logits.dataPtr(), logits.numel(), null, 0);
        } finally {
            logits.close();
        }
    }

    static int sampleGreedyFromTensor(AccelTensor logits, ModelConfig config, Tokenizer tokenizer,
            boolean firstStep, Set<Integer> stops) {
        return sampleGreedyFromTensor(logits, config, tokenizer, null, firstStep, stops, null);
    }

    static int sampleGreedyFromTensor(AccelTensor logits, ModelConfig config, Tokenizer tokenizer,
            ModelRuntimeTraits traits, boolean firstStep, Set<Integer> stops, TokenSamplingMasks masks) {
        return sampleGreedyFromTensor(logits, config, tokenizer, traits, firstStep, stops, masks, null);
    }

    static int sampleGreedyFromTensor(AccelTensor logits, ModelConfig config, Tokenizer tokenizer,
            ModelRuntimeTraits traits, boolean firstStep, Set<Integer> stops, TokenSamplingMasks masks,
            GenerationTokenValidationCache validationCache) {
        try {
            ModelRuntimeTraits effectiveTraits = effectiveTraits(config, traits);
            MemorySegment seg = logits.dataPtr();
            long vocab = logits.numel();
            Double cap = config.finalLogitSoftcapping();
            float softCap = cap != null && cap > 0 ? cap.floatValue() : 0.0f;
            BitSet baseMask = masks == null
                    ? GenerationTokenValidationPolicy.buildDisallowedTokenMask(
                            tokenizer, firstStep, stops, (int) vocab, effectiveTraits)
                    : masks.maskFor(firstStep);
            boolean needsTokenizerValidation = firstStep || effectiveTraits.validateContinuationTokensByDecode();
            int[] rejectedCandidates = null;
            int rejectedCount = 0;

            for (int attempt = 0; attempt < GREEDY_RETRY_LIMIT; attempt++) {
                if (attempt == 0) {
                    GenerationTokenDebug.topGreedyCandidates(seg, vocab, softCap, tokenizer, effectiveTraits,
                            firstStep,
                            stops, 8);
                }
                int best = GenerationGreedyArgmax.select(seg, vocab, rejectedCandidates, rejectedCount);
                if (best < 0) {
                    break;
                }
                boolean masked = baseMask != null && baseMask.get(best);
                if (!masked && (!needsTokenizerValidation
                        || !shouldRejectSampledToken(best, tokenizer, effectiveTraits, firstStep, stops,
                                validationCache))) {
                    GenerationTokenDebug.sampleChoice("accept", best, tokenizer, firstStep, attempt);
                    return best;
                }
                GenerationTokenDebug.sampleChoice("reject", best, tokenizer, firstStep, attempt);
                if (rejectedCandidates == null) {
                    rejectedCandidates = new int[GREEDY_REJECTED_CANDIDATE_CAPACITY];
                }
                if (rejectedCount < rejectedCandidates.length) {
                    rejectedCandidates[rejectedCount++] = best;
                }
            }
            return selectWithFullMaskAfterCheapRetries(
                    seg, vocab, baseMask, rejectedCandidates, rejectedCount, tokenizer, effectiveTraits, firstStep,
                    stops, validationCache, needsTokenizerValidation);
        } finally {
            logits.close();
        }
    }

    static boolean isGemma4Text(ModelConfig config) {
        return ModelRuntimeTraitsResolver.resolve(config).nativeBf16Matvec();
    }

    static EncodeOptions encodeOptionsFor(ModelConfig config) {
        return encodeOptionsFor(config, null, null);
    }

    static EncodeOptions encodeOptionsFor(ModelConfig config, String prompt) {
        return encodeOptionsFor(config, null, prompt);
    }

    static EncodeOptions encodeOptionsFor(ModelConfig config, ModelRuntimeTraits traits, String prompt) {
        EncodeOptions options = EncodeOptions.defaultOptions();
        ModelRuntimeTraits effectiveTraits = effectiveTraits(config, traits);
        switch (effectiveTraits.promptBosPolicy()) {
            case NEVER -> options.addBos = false;
            case TURN_AWARE -> options.addBos = !looksLikeTurnPrompt(prompt, effectiveTraits.turnPromptPrefixes());
            case DEFAULT -> {
            }
        }
        return options;
    }

    private static ModelRuntimeTraits effectiveTraits(ModelConfig config, ModelRuntimeTraits traits) {
        return ModelRuntimeTraitsResolver.resolve(config, traits);
    }

    private static boolean hasMaskedCandidates(BitSet mask) {
        return mask != null && !mask.isEmpty();
    }

    private static int selectWithFullMaskAfterCheapRetries(
            MemorySegment seg,
            long vocab,
            BitSet baseMask,
            int[] rejectedCandidates,
            int rejectedCount,
            Tokenizer tokenizer,
            ModelRuntimeTraits traits,
            boolean firstStep,
            Set<Integer> stops,
            GenerationTokenValidationCache validationCache,
            boolean needsTokenizerValidation) {
        if (!hasMaskedCandidates(baseMask) && rejectedCount < GREEDY_REJECTED_CANDIDATE_CAPACITY) {
            return -1;
        }
        BitSet fullMask = combinedRejectedMask(baseMask, rejectedCandidates, rejectedCount);
        for (int offset = 0; offset < GREEDY_FULL_MASK_RETRY_LIMIT; offset++) {
            int attempt = GREEDY_RETRY_LIMIT + offset;
            int best = GenerationGreedyArgmax.selectWithMask(seg, vocab, fullMask, null, 0);
            if (best < 0) {
                break;
            }
            if (!needsTokenizerValidation
                    || !shouldRejectSampledToken(best, tokenizer, traits, firstStep, stops, validationCache)) {
                GenerationTokenDebug.sampleChoice("accept", best, tokenizer, firstStep, attempt);
                return best;
            }
            GenerationTokenDebug.sampleChoice("reject", best, tokenizer, firstStep, attempt);
            fullMask.set(best);
        }
        return -1;
    }

    private static BitSet combinedRejectedMask(BitSet baseMask, int[] rejectedCandidates, int rejectedCount) {
        BitSet mask = baseMask == null ? new BitSet() : (BitSet) baseMask.clone();
        if (rejectedCandidates == null) {
            return mask;
        }
        for (int i = 0; i < rejectedCount; i++) {
            int tokenId = rejectedCandidates[i];
            if (tokenId >= 0) {
                mask.set(tokenId);
            }
        }
        return mask;
    }

    private static void maskDisallowedContinuationTokens(float[] logits, Tokenizer tokenizer, boolean firstStep,
            Set<Integer> stops, ModelRuntimeTraits traits, TokenSamplingMasks masks) {
        if (masks != null) {
            GenerationTokenValidationPolicy.maskDisallowedContinuationTokens(logits, masks.maskFor(firstStep));
            return;
        }
        GenerationTokenValidationPolicy.maskDisallowedContinuationTokens(logits, tokenizer, firstStep, stops, traits);
    }

    private static boolean shouldRejectSampledToken(int tokenId, Tokenizer tokenizer, ModelRuntimeTraits traits,
            boolean firstStep, Set<Integer> stops, GenerationTokenValidationCache validationCache) {
        if (validationCache != null) {
            return validationCache.shouldRejectSampledToken(tokenId, tokenizer, traits, firstStep, stops);
        }
        return GenerationTokenValidationPolicy.shouldRejectSampledToken(tokenId, tokenizer, traits, firstStep, stops);
    }

    private static boolean looksLikeTurnPrompt(String prompt, Set<String> turnPrefixes) {
        if (prompt == null || turnPrefixes == null || turnPrefixes.isEmpty()) {
            return false;
        }
        String trimmed = prompt.stripLeading();
        if (trimmed.startsWith("<bos>")) {
            trimmed = trimmed.substring("<bos>".length()).stripLeading();
        }
        for (String prefix : turnPrefixes) {
            if (trimmed.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }
}
