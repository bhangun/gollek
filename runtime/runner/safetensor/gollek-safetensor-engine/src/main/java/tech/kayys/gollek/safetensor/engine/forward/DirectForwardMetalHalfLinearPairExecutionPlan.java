/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardNativeBf16MatvecPolicy.describeNativeBf16PairMatvecPath;

record DirectForwardMetalHalfLinearPairExecutionPlan(
        int rows,
        int inputDim,
        int outputDim,
        boolean nativeBf16Weights,
        boolean nativeBf16MatvecCandidate,
        boolean halfMatvecCandidate) {

    static DirectForwardMetalHalfLinearPairExecutionPlan from(
            int rows,
            int inputDim,
            int outputDim,
            boolean nativeBf16Weights,
            DirectForwardMetalCapabilities capabilities,
            boolean pairMatvecAllowed) {
        boolean singleRow = rows == 1;
        return new DirectForwardMetalHalfLinearPairExecutionPlan(
                rows,
                inputDim,
                outputDim,
                nativeBf16Weights,
                singleRow
                        && nativeBf16Weights
                        && pairMatvecAllowed
                        && capabilities.supportsMatvecTransposedRightBf16Pair(),
                singleRow
                        && !nativeBf16Weights
                        && pairMatvecAllowed
                        && capabilities.supportsMatvecTransposedRightHalfPair());
    }

    String nativeBf16MatvecPath() {
        return describeNativeBf16PairMatvecPath(inputDim, outputDim);
    }

    String halfMatvecPath() {
        return "matvec";
    }

    String matmulPath() {
        return "metal_pair_matmul";
    }
}
