/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

final class GenerationTokenValidationCache {
    private static final String ARRAY_MAX_VOCAB_PROPERTY =
            "gollek.safetensor.validation_cache_array_max_vocab";
    private static final int DEFAULT_ARRAY_MAX_VOCAB = 1_000_000;
    private static final byte UNKNOWN = 0;
    private static final byte ACCEPTED = 1;
    private static final byte REJECTED = 2;

    private final byte[] firstStepVerdicts;
    private final byte[] continuationVerdicts;
    private final Map<Integer, Boolean> firstStepRejects;
    private final Map<Integer, Boolean> continuationRejects;
    private Tokenizer specialTokenTokenizer;
    private Map<Integer, String> specialTokenTexts;

    GenerationTokenValidationCache() {
        this(-1);
    }

    GenerationTokenValidationCache(int vocabSize) {
        if (canUseArrayCache(vocabSize)) {
            firstStepVerdicts = new byte[vocabSize];
            continuationVerdicts = new byte[vocabSize];
            firstStepRejects = null;
            continuationRejects = null;
        } else {
            firstStepVerdicts = null;
            continuationVerdicts = null;
            firstStepRejects = new HashMap<>();
            continuationRejects = new HashMap<>();
        }
    }

    boolean shouldRejectSampledToken(int tokenId, Tokenizer tokenizer, ModelRuntimeTraits traits,
            boolean firstStep, Set<Integer> stops) {
        if (tokenId < 0) {
            return false;
        }
        byte[] verdicts = verdictsFor(firstStep);
        if (verdicts != null) {
            if (tokenId >= verdicts.length) {
                return computeRejection(tokenId, tokenizer, traits, firstStep, stops);
            }
            byte cached = verdicts[tokenId];
            if (cached != UNKNOWN) {
                return cached == REJECTED;
            }
            boolean rejected = computeRejection(tokenId, tokenizer, traits, firstStep, stops);
            verdicts[tokenId] = rejected ? REJECTED : ACCEPTED;
            return rejected;
        }
        Map<Integer, Boolean> cache = firstStep ? firstStepRejects : continuationRejects;
        Boolean cached = cache.get(tokenId);
        if (cached != null) {
            return cached;
        }
        boolean rejected = computeRejection(tokenId, tokenizer, traits, firstStep, stops);
        cache.put(tokenId, rejected);
        return rejected;
    }

    private byte[] verdictsFor(boolean firstStep) {
        return firstStep ? firstStepVerdicts : continuationVerdicts;
    }

    private boolean computeRejection(int tokenId, Tokenizer tokenizer, ModelRuntimeTraits traits,
            boolean firstStep, Set<Integer> stops) {
        return GenerationTokenValidationPolicy.shouldRejectSampledToken(
                tokenId, tokenizer, traits, firstStep, stops, specialTokenTexts(tokenizer));
    }

    private Map<Integer, String> specialTokenTexts(Tokenizer tokenizer) {
        if (tokenizer == null) {
            return Map.of();
        }
        if (specialTokenTexts != null && specialTokenTokenizer == tokenizer) {
            return specialTokenTexts;
        }
        Map<String, Integer> tokens = tokenizer.specialTokens();
        if (tokens == null || tokens.isEmpty()) {
            specialTokenTokenizer = tokenizer;
            specialTokenTexts = Map.of();
            return specialTokenTexts;
        }
        Map<Integer, String> byId = new LinkedHashMap<>(tokens.size());
        for (Map.Entry<String, Integer> entry : tokens.entrySet()) {
            Integer id = entry.getValue();
            if (id != null && id >= 0) {
                byId.putIfAbsent(id, entry.getKey());
            }
        }
        specialTokenTokenizer = tokenizer;
        specialTokenTexts = Map.copyOf(byId);
        return specialTokenTexts;
    }

    private static boolean canUseArrayCache(int vocabSize) {
        return vocabSize > 0 && vocabSize <= maxArrayCacheVocab();
    }

    private static int maxArrayCacheVocab() {
        return Integer.getInteger(ARRAY_MAX_VOCAB_PROPERTY, DEFAULT_ARRAY_MAX_VOCAB);
    }
}
