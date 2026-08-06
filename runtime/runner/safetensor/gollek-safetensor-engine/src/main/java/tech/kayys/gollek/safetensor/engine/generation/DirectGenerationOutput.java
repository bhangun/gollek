/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import java.util.List;

/**
 * Response-facing view of a completed decode loop.
 */
record DirectGenerationOutput(String text, long[] generatedTokenIds, int completionTokens) {
    private static final DirectGenerationOutput EMPTY = new DirectGenerationOutput("", new long[0], 0);

    static DirectGenerationOutput empty() {
        return EMPTY;
    }

    static DirectGenerationOutput orEmpty(DirectGenerationOutput output) {
        return output == null ? empty() : output;
    }

    static DirectGenerationOutput fromLoop(DirectGenerationLoop.Result loop) {
        if (loop == null) {
            return empty();
        }
        return new DirectGenerationOutput(
                loop.text(),
                toLongArray(loop.generatedTokenIds()),
                loop.completionTokens());
    }

    static DirectGenerationOutput fromLoop(DirectGenerationLoop.Result loop, DirectGenerationTimings timings) {
        if (timings != null) {
            timings.recordLoop(loop);
        }
        return fromLoop(loop);
    }

    int generatedTokenCount() {
        return generatedTokenIds.length;
    }

    private static long[] toLongArray(List<Long> values) {
        if (values == null || values.isEmpty()) {
            return new long[0];
        }
        long[] out = new long[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }
}
