/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.isSingleTokenHalfLinearCandidate;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

final class DirectForwardLinearCachePolicy {
    private static final DirectForwardLinearCacheOptions OPTIONS =
            DirectForwardLinearCacheOptions.fromSystemProperties();
    private static final DirectForwardLinearCacheRoutingPolicy ROUTING =
            DirectForwardLinearCacheRoutingPolicy.from(OPTIONS);

    private DirectForwardLinearCachePolicy() {
    }

    static AccelTensor cachedFfnDownHalfWeight(AccelTensor input, AccelTensor weight,
            ModelConfig config, String profileKey) {
        if (!ROUTING.isFfnDownHalfCacheProfile(profileKey)) {
            return null;
        }
        if (!ROUTING.shouldCacheFfnDownHalfWeight(
                profileKey,
                isSingleTokenHalfLinearCandidate(input, weight),
                weight.dequantizedByteSize(),
                config.numHiddenLayers())) {
            return null;
        }
        // Do not preemptively dequantize to avoid SIGILL on unaligned quantized tensors.
        return null;
    }

    static AccelTensor cachedLogitsLargeHalfWeight(AccelTensor input, AccelTensor weight, String profileKey) {
        return null;
    }


    static AccelTensor toMetalHalfWeight(AccelTensor weight, boolean nativeBf16, boolean nativeBf16Matvec,
            boolean allowBf16ToF16, boolean allowMetalBf16Linear) {
        if (weight == null) {
            return null;
        }
        if (weight.quantType() == AccelTensor.QuantType.F16) {
            return weight;
        }
        if (nativeBf16 && weight.quantType() == AccelTensor.QuantType.BF16) {
            return weight;
        }
        if (weight.quantType() == AccelTensor.QuantType.BF16 && nativeBf16Matvec) {
            return allowBf16ToF16
                    ? weight.toF16CachedUpTo(OPTIONS.metalF16WeightCacheMaxBytes())
                    : null;
        }
        if (weight.quantType() == AccelTensor.QuantType.BF16 && allowMetalBf16Linear) {
            return weight.toF16CachedUpTo(OPTIONS.metalF16WeightCacheMaxBytes());
        }
        if (weight.quantType() == AccelTensor.QuantType.NF4) {
            return weight;
        }
        if (weight.isQuantized()) {
            // Do NOT preemptively dequantize on CPU to avoid SIGILL and memory bloat.
            // Native Metal plugins (e.g. TurboQuant) will handle the quantized tensors directly.
            return weight;
        }
        return null;
    }


    static AccelTensor cachedTransposedF16Weight(AccelTensor weight) {
        return weight.toF16Transposed2dCachedUpTo(OPTIONS.metalF16WeightCacheMaxBytes());
    }
}
