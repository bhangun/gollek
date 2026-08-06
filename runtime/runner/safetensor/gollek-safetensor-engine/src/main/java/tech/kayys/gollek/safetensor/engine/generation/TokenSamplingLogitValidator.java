/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.util.Arrays;

final class TokenSamplingLogitValidator {
    private static final String VALIDATE_LOGITS_PROPERTY = "gollek.safetensor.validate_logits";

    private TokenSamplingLogitValidator() {
    }

    static boolean accepts(float[] logits) {
        if (logits == null || logits.length == 0) {
            System.err.println("ERROR: Empty or null logits array");
            return false;
        }

        return !Boolean.getBoolean(VALIDATE_LOGITS_PROPERTY) || validate(logits);
    }

    static boolean validate(float[] logits) {
        int nanCount = 0;
        int infCount = 0;
        float minVal = Float.MAX_VALUE;
        float maxVal = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            if (Float.isNaN(logit)) {
                nanCount++;
            }
            if (Float.isInfinite(logit)) {
                infCount++;
            }
            if (logit < minVal) {
                minVal = logit;
            }
            if (logit > maxVal) {
                maxVal = logit;
            }
        }

        if (nanCount > logits.length * 0.1 || infCount > logits.length * 0.1) {
            System.err.println("WARNING: Corrupted logits detected!");
            System.err.println("  NaN count: " + nanCount + "/" + logits.length);
            System.err.println("  Inf count: " + infCount + "/" + logits.length);
            System.err.println("  Range: [" + minVal + ", " + maxVal + "]");
            System.err.println("  First 10 logits: " + Arrays.toString(
                    Arrays.copyOf(logits, Math.min(10, logits.length))));
            return false;
        }
        return true;
    }
}
