/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalLinearShapePlan(
        long inputDim,
        long rows,
        long outputDim,
        long[] outputShape) {

    static DirectForwardMetalLinearShapePlan single(AccelTensor input, AccelTensor weight) {
        if (input == null || weight == null || input.rank() < 2 || weight.rank() != 2) {
            return null;
        }
        long inputDim = input.size(-1);
        if (inputDim <= 0L || weight.size(1) != inputDim) {
            return null;
        }
        long rows = input.numel() / inputDim;
        if (rows <= 0L) {
            return null;
        }
        long outputDim = weight.size(0);
        if (outputDim <= 0L) {
            return null;
        }
        return new DirectForwardMetalLinearShapePlan(inputDim, rows, outputDim,
                input.shapeWithLastDim(outputDim));
    }

    static DirectForwardMetalLinearShapePlan pair(
            AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor secondWeight) {
        DirectForwardMetalLinearShapePlan plan = single(input, firstWeight);
        if (plan == null || secondWeight == null || secondWeight.rank() != 2) {
            return null;
        }
        if (secondWeight.size(0) != plan.outputDim() || secondWeight.size(1) != plan.inputDim()) {
            return null;
        }
        return plan;
    }

    boolean singleRow() {
        return rows == 1L;
    }
}
