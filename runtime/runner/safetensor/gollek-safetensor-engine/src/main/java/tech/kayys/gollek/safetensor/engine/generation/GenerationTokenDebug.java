/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Set;

final class GenerationTokenDebug {
    private GenerationTokenDebug() {
    }

    static void tokenSequence(Tokenizer tokenizer, long[] tokenIds, String label) {
        System.out.printf("[DEBUG-TOKENS] %s:", label);
        for (long tokenId : tokenIds) {
            String decoded = GenerationTokenValidationPolicy.decodeSingleToken(tokenizer, (int) tokenId, false);
            String printable = printableText(decoded);
            System.out.printf(" [%d:%s]", tokenId, printable);
        }
        System.out.println();
        System.out.flush();
    }

    static String printableText(String text) {
        if (text == null) {
            return "<null>";
        }
        return text.replace("\n", "\\n");
    }

    static void chosenToken(Tokenizer tokenizer, int tokenId, int step) {
        if (!Boolean.getBoolean("gollek.verbose")) {
            return;
        }
        String decoded = GenerationTokenValidationPolicy.decodeSingleToken(tokenizer, tokenId, false);
        System.out.printf("[DEBUG-CHOSEN] step=%d token=%d text=%s%n",
                step, tokenId, printableText(decoded));
        System.out.flush();
    }

    static void sampleChoice(String decision, int tokenId, Tokenizer tokenizer, boolean firstStep, int attempt) {
        if (!Boolean.getBoolean("gollek.verbose")) {
            return;
        }
        String decoded = GenerationTokenValidationPolicy.decodeSingleToken(tokenizer, tokenId, false);
        System.out.printf("[DEBUG-SAMPLE] %s firstStep=%s attempt=%d token=%d text=%s%n",
                decision, firstStep, attempt, tokenId, printableText(decoded));
        System.out.flush();
    }

    static void topGreedyCandidates(MemorySegment seg,
            long vocab,
            float softCap,
            Tokenizer tokenizer,
            ModelRuntimeTraits traits,
            boolean firstStep,
            Set<Integer> stops,
            int limit) {
        if (!Boolean.getBoolean("gollek.verbose")) {
            return;
        }
        int topN = Math.max(1, limit);
        int[] ids = new int[topN];
        float[] vals = new float[topN];
        Arrays.fill(ids, -1);
        Arrays.fill(vals, Float.NEGATIVE_INFINITY);

        for (int i = 0; i < vocab; i++) {
            float value = applyLogitSoftcap(seg.getAtIndex(ValueLayout.JAVA_FLOAT, i), softCap);
            for (int slot = 0; slot < topN; slot++) {
                if (value > vals[slot]) {
                    for (int shift = topN - 1; shift > slot; shift--) {
                        vals[shift] = vals[shift - 1];
                        ids[shift] = ids[shift - 1];
                    }
                    vals[slot] = value;
                    ids[slot] = i;
                    break;
                }
            }
        }

        System.out.printf("[DEBUG-TOP] firstStep=%s%n", firstStep);
        for (int rank = 0; rank < topN; rank++) {
            int tokenId = ids[rank];
            if (tokenId < 0) {
                continue;
            }
            boolean disallowed = GenerationTokenValidationPolicy.isDisallowedContinuationToken(
                    tokenId, tokenizer, firstStep, stops, traits);
            boolean rejected = GenerationTokenValidationPolicy.shouldRejectSampledToken(
                    tokenId, tokenizer, traits, firstStep, stops);
            String decoded = GenerationTokenValidationPolicy.decodeSingleToken(tokenizer, tokenId, false);
            System.out.printf("[DEBUG-TOP] rank=%d token=%d logit=%f disallowed=%s rejected=%s text=%s%n",
                    rank + 1, tokenId, vals[rank], disallowed, rejected, printableText(decoded));
        }
        System.out.flush();
    }

    private static float applyLogitSoftcap(float value, float softCap) {
        if (softCap <= 0.0f) {
            return value;
        }
        return (float) (softCap * Math.tanh(value / softCap));
    }
}
