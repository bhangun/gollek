/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Arrays;

/**
 * Validates attention projection shapes before reshaping into head layout.
 */
final class FlashAttentionShapeValidator {
    private FlashAttentionShapeValidator() {
    }

    static AccelTensor reshapeProjection(AccelTensor projection, String label, long batch, long seqLen,
            int heads, int headDim, ModelConfig config, int layerIdx) {
        validateProjection(projection, label, batch, seqLen, heads, headDim, config, layerIdx);
        return projection.reshape(batch, seqLen, heads, headDim);
    }

    static void validateProjection(AccelTensor projection, String label, int heads, int headDim,
            ModelConfig config, int layerIdx) {
        validateProjection(projection, label, -1, -1, heads, headDim, config, layerIdx);
    }

    static void validateProjection(AccelTensor projection, String label, long batch, long seqLen,
            int heads, int headDim, ModelConfig config, int layerIdx) {
        if (projection == null) {
            throw new IllegalArgumentException("Missing attention " + label + " projection at layer " + layerIdx);
        }
        long expectedLastDim = (long) heads * headDim;
        long actualLastDim = projection.size(-1);
        long expectedElements = expectedElements(batch, seqLen, expectedLastDim);
        long actualElements = projection.numel();
        if (actualLastDim == expectedLastDim
                && (expectedElements < 0 || actualElements == expectedElements)) {
            return;
        }
        String modelType = config == null ? "<unknown>" : config.modelType();
        int hiddenSize = config == null ? 0 : config.hiddenSize();
        throw new IllegalArgumentException(
                "Attention " + label + " projection shape mismatch at layer " + layerIdx
                        + ": lastDim=" + actualLastDim
                        + " expected=" + expectedLastDim
                        + " heads=" + heads
                        + " headDim=" + headDim
                        + " modelType=" + modelType
                        + " hiddenSize=" + hiddenSize
                        + targetShapeMessage(batch, seqLen, heads, headDim, expectedElements, actualElements)
                        + " shape=" + Arrays.toString(projection.shape()));
    }

    private static long expectedElements(long batch, long seqLen, long expectedLastDim) {
        if (batch < 0 || seqLen < 0) {
            return -1;
        }
        return Math.multiplyExact(Math.multiplyExact(batch, seqLen), expectedLastDim);
    }

    private static String targetShapeMessage(long batch, long seqLen, int heads, int headDim,
            long expectedElements, long actualElements) {
        if (expectedElements < 0) {
            return " actualElements=" + actualElements;
        }
        return " targetShape=" + Arrays.toString(new long[] { batch, seqLen, heads, headDim })
                + " actualElements=" + actualElements
                + " expectedElements=" + expectedElements;
    }
}
