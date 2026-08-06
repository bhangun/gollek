/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalFfnShapePlan(
        long inputDim,
        long rows,
        long intermediateDim,
        long outputDim,
        long[] outputShape) {

    static DirectForwardMetalFfnShapePlan gated(
            AccelTensor input,
            AccelTensor gateW,
            AccelTensor upW,
            AccelTensor downW) {
        if (input == null) {
            return null;
        }
        long inputDim = input.size(-1);
        if (!validGatedWeights(gateW, upW, downW, inputDim)) {
            return null;
        }
        long rows = rows(input, inputDim);
        long outputDim = downW.size(0);
        return new DirectForwardMetalFfnShapePlan(
                inputDim,
                rows,
                gateW.size(0),
                outputDim,
                input.shapeWithLastDim(outputDim));
    }

    static DirectForwardMetalFfnShapePlan gateUp(
            AccelTensor input,
            AccelTensor gateW,
            AccelTensor upW) {
        if (input == null) {
            return null;
        }
        long inputDim = input.size(-1);
        if (!validGateUpWeights(gateW, upW, inputDim)) {
            return null;
        }
        long rows = rows(input, inputDim);
        long intermediateDim = gateW.size(0);
        return new DirectForwardMetalFfnShapePlan(
                inputDim,
                rows,
                intermediateDim,
                intermediateDim,
                input.shapeWithLastDim(intermediateDim));
    }

    static long rows(AccelTensor input) {
        return rows(input, input.size(-1));
    }

    boolean singleRow() {
        return rows == 1L;
    }

    boolean validRows() {
        return rows > 0L;
    }

    boolean matchesOutputBuffer(AccelTensor outputBuffer) {
        return outputBuffer != null && outputBuffer.hasShape(outputShape);
    }

    private static boolean validGatedWeights(
            AccelTensor gateW,
            AccelTensor upW,
            AccelTensor downW,
            long inputDim) {
        return validGateUpWeights(gateW, upW, inputDim)
                && downW != null
                && downW.rank() == 2
                && downW.size(1) == gateW.size(0);
    }

    private static boolean validGateUpWeights(
            AccelTensor gateW,
            AccelTensor upW,
            long inputDim) {
        return gateW != null
                && upW != null
                && gateW.rank() == 2
                && upW.rank() == 2
                && gateW.size(0) == upW.size(0)
                && gateW.size(1) == upW.size(1)
                && gateW.size(1) == inputDim;
    }

    private static long rows(AccelTensor input, long inputDim) {
        return input.numel() / Math.max(1L, inputDim);
    }
}
