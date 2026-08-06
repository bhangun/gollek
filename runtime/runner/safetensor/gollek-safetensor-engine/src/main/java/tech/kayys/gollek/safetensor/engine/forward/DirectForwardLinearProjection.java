/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelOps;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;

final class DirectForwardLinearProjection {
    private DirectForwardLinearProjection() {
    }

    static AccelTensor ffnDownLinear(DirectForwardLinearRequest request) {
        if (request.runtime().metalLinearEnabled()) {
            return linear(request);
        }
        AccelTensor cached = tryCachedFfnDownHalfLinear(request);
        if (cached != null) {
            return cached;
        }
        return linear(request);
    }

    static AccelTensor linear(DirectForwardLinearRequest request) {
        String profileKey = request.profileKey();
        long t0 = profileKey != null ? System.nanoTime() : 0L;
        boolean metalLinearEnabled = request.runtime().metalLinearEnabled();
        if (request.weight().isQuantized()
                && request.weight().quantType() != AccelTensor.QuantType.BF16
                && request.weight().quantType() != AccelTensor.QuantType.F16
                && request.weight().quantType() != AccelTensor.QuantType.NF4
                && request.weight().quantType() != AccelTensor.QuantType.INT4) {
            AccelTensor dequantized = request.weight().dequantize();
            try {
                return linear(request.withWeight(dequantized));
            } finally {
                if (dequantized != request.weight() && !dequantized.isClosed()) {
                    dequantized.close();
                }
            }
        }

        AccelTensor metalHalf = DirectForwardMetalHalfLinear.tryLinear(
                request.runtime().log(),
                request.runtime().metalBinding(),
                request.runtime().capabilities(),
                request.traits(),
                request.config(),
                metalLinearEnabled,
                request.decodeLogitsPhase(),
                request.input(),
                request.weight(),
                request.bias(),
                request.outputBuffer(),
                profileKey);
        if (metalHalf != null) {
            if (profileKey != null) {
                DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
            }
            return metalHalf;
        }

        AccelTensor metalFloat = DirectForwardMetalFloatLinear.tryLinear(
                request.runtime().log(),
                request.runtime().metalBinding(),
                metalLinearEnabled,
                request.input(),
                request.weight(),
                request.bias());
        if (metalFloat != null) {
            if (profileKey != null) {
                DirectInferenceProfiler.recordLinearPath(profileKey, "metal_float_matmul");
                DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
            }
            return metalFloat;
        }

        AccelTensor cachedLargeHalf = tryCachedLargeHalfLinear(request);
        if (cachedLargeHalf != null) {
            if (profileKey != null) {
                DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
            }
            return cachedLargeHalf;
        }

        AccelTensor mm = AccelOps.linear(request.input(), request.weight());
        if (request.bias() != null) {
            AccelTensor out = AccelOps.add(mm, request.bias());
            mm.close();
            if (profileKey != null) {
                DirectInferenceProfiler.recordLinearPath(profileKey, "accelerate_linear_with_bias");
                DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
            }
            return out;
        }
        if (profileKey != null) {
            DirectInferenceProfiler.recordLinearPath(profileKey, "accelerate_linear");
            DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
        }
        return mm;
    }

    private static AccelTensor tryCachedFfnDownHalfLinear(DirectForwardLinearRequest request) {
        AccelTensor dequantized = DirectForwardLinearCachePolicy.cachedFfnDownHalfWeight(
                request.input(),
                request.weight(),
                request.config(),
                request.profileKey());
        if (dequantized == null) {
            return null;
        }
        return linear(request.withFallbackContext(dequantized));
    }

    private static AccelTensor tryCachedLargeHalfLinear(DirectForwardLinearRequest request) {
        AccelTensor dequantized = DirectForwardLinearCachePolicy.cachedLogitsLargeHalfWeight(
                request.input(),
                request.weight(),
                request.profileKey());
        if (dequantized == null) {
            return null;
        }
        return linear(request.withFallbackContext(dequantized));
    }
}
