/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;

record DirectForwardMetalCapabilities(
        boolean nativeElementwiseKernelsAvailable,
        boolean nativeElementwiseFallbackAvailable,
        boolean nativeScaleKernelAvailable,
        boolean supportsSwigluFfnHalf,
        boolean supportsGegluFfnHalf,
        boolean supportsSwigluFfnMatvecHalf,
        boolean supportsGegluFfnMatvecHalf,
        boolean supportsSwigluFfnMatvecBf16,
        boolean supportsGegluFfnMatvecBf16,
        boolean supportsSwigluFfnMatvecRowsBf16,
        boolean supportsGegluFfnMatvecRowsBf16,
        boolean supportsSwigluGateUpMatvecHalf,
        boolean supportsGegluGateUpMatvecHalf,
        boolean supportsSwigluGateUpMatvecBf16,
        boolean supportsGegluGateUpMatvecBf16,
        boolean supportsMatmulTransposedRightHalfPair,
        boolean supportsMatvecTransposedRightHalfPair,
        boolean supportsMatvecTransposedRightBf16Pair,
        boolean supportsMatvecTransposedRightHalf,
        boolean supportsMatvecTransposedRightHalfMps,
        boolean supportsMatvecTransposedRightBf16,
        boolean supportsMatvecTransposedRightNf4,
        boolean supportsMatvecTransposedRightInt4,
        boolean supportsMatvecTransposedWeightHalf) {

    static final DirectForwardMetalCapabilities EMPTY = new DirectForwardMetalCapabilities(
            false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false,
            false, false, false, false);

    static DirectForwardMetalCapabilities detect(MetalBinding binding) {
        if (binding == null) {
            return EMPTY;
        }
        try {
            return new DirectForwardMetalCapabilities(
                    binding.nativeElementwiseKernelsAvailable(),
                    binding.nativeElementwiseFallbackAvailable(),
                    binding.nativeScaleKernelAvailable(),
                    binding.supportsSwigluFfnHalf(),
                    binding.supportsGegluFfnHalf(),
                    binding.supportsSwigluFfnMatvecHalf(),
                    binding.supportsGegluFfnMatvecHalf(),
                    binding.supportsSwigluFfnMatvecBf16(),
                    binding.supportsGegluFfnMatvecBf16(),
                    binding.supportsSwigluFfnMatvecRowsBf16(),
                    binding.supportsGegluFfnMatvecRowsBf16(),
                    binding.supportsSwigluGateUpMatvecHalf(),
                    binding.supportsGegluGateUpMatvecHalf(),
                    binding.supportsSwigluGateUpMatvecBf16(),
                    binding.supportsGegluGateUpMatvecBf16(),
                    binding.supportsMatmulTransposedRightHalfPair(),
                    binding.supportsMatvecTransposedRightHalfPair(),
                    binding.supportsMatvecTransposedRightBf16Pair(),
                    binding.supportsMatvecTransposedRightHalf(),
                    binding.supportsMatvecTransposedRightHalfMps(),
                    binding.supportsMatvecTransposedRightBf16(),
                    binding.supportsMatvecTransposedRightNf4(),
                    binding.supportsMatvecTransposedRightInt4(),
                    binding.supportsMatvecTransposedWeightHalf());
        } catch (Exception e) {
            return EMPTY;
        }
    }
}
