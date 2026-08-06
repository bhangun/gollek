/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

/**
 * Describes flattened matrix dimensions for Metal attention linear kernels and
 * validates reusable output buffers before backend writes.
 */
final class FlashAttentionMetalLinearPlan {
    private final long[] inputShape;
    private final long rows;
    private final long k;
    private final long[] outputDims;

    private FlashAttentionMetalLinearPlan(long[] inputShape, long rows, long k, long[] outputDims) {
        this.inputShape = inputShape;
        this.rows = rows;
        this.k = k;
        this.outputDims = outputDims;
    }

    static FlashAttentionMetalLinearPlan resolveInput(AccelTensor input) {
        if (input == null) {
            return null;
        }
        long[] inputShape = input.shape();
        if (inputShape.length < 2) {
            return null;
        }

        long k = inputShape[inputShape.length - 1];
        long rows = input.numel() / Math.max(1L, k);
        if (rows <= 0L) {
            return null;
        }

        long batchProduct = 1L;
        for (int i = 0; i < inputShape.length - 2; i++) {
            batchProduct *= inputShape[i];
        }
        if (batchProduct != 1L) {
            return null;
        }

        return new FlashAttentionMetalLinearPlan(inputShape, rows, k, new long[0]);
    }

    static FlashAttentionMetalLinearPlan resolve(AccelTensor input, AccelTensor... weights) {
        FlashAttentionMetalLinearPlan inputPlan = resolveInput(input);
        return inputPlan == null ? null : inputPlan.withWeights(weights);
    }

    FlashAttentionMetalLinearPlan withWeights(AccelTensor... weights) {
        if (weights == null || weights.length == 0) {
            return this;
        }
        long[] dims = new long[weights.length];
        for (int i = 0; i < weights.length; i++) {
            AccelTensor weight = weights[i];
            if (weight == null) {
                return null;
            }
            long[] weightShape = weight.shape();
            if (weightShape.length != 2 || weightShape[1] != k) {
                return null;
            }
            dims[i] = weightShape[0];
        }
        return new FlashAttentionMetalLinearPlan(inputShape, rows, k, dims);
    }

    int m() {
        return Math.toIntExact(rows);
    }

    int kk() {
        return Math.toIntExact(k);
    }

    int n(int index) {
        return Math.toIntExact(outputDims[index]);
    }

    long k() {
        return k;
    }

    long outputDim(int index) {
        return outputDims[index];
    }

    long[] outputShape(int index) {
        long[] outputShape = inputShape.clone();
        outputShape[outputShape.length - 1] = outputDims[index];
        return outputShape;
    }

    AccelTensor reusableOutputTensor(int index, AccelTensor outputBuffer) {
        long[] outputShape = outputShape(index);
        if (outputBuffer != null
                && !outputBuffer.isClosed()
                && outputBuffer.quantType() == AccelTensor.QuantType.F32
                && outputBuffer.isContiguous()
                && outputBuffer.hasShape(outputShape)
                && outputBuffer.dataPtr().byteSize() >= Math.multiplyExact(outputBuffer.numel(), (long) Float.BYTES)) {
            return outputBuffer;
        }
        return AccelTensor.zeros(outputShape);
    }
}
