/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelOps;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.alkhawarizm.spi.model.FFNActivationType;

final class DirectForwardLocalFusedGatedFfn {
    private static final String PROFILE_KEY = "ffn_gate_up_fused";
    private static final String GATE_UP_PAIR_PROFILE_KEY = "ffn_gate_up_pair";

    private DirectForwardLocalFusedGatedFfn() {
    }

    static AccelTensor tryFfn(ModelConfigTraits traits,
                              boolean metalLinearEnabled,
                              AccelTensor input,
                              FFNActivationType activationType,
                              AccelTensor gateW,
                              AccelTensor gateB,
                              AccelTensor upW,
                              AccelTensor upB,
                              AccelTensor outputBuffer) {
        if (!DirectForwardFfnFastPathPolicy.shouldTryLocalFusedHalfFfn(traits)
                || shouldPreferSeparateMetalHalfFfn(traits, metalLinearEnabled, input, gateW, upW)) {
            return null;
        }
        if (gateW == null || upW == null || gateW.rank() != 2 || upW.rank() != 2) {
            return null;
        }
        long outputDim = gateW.size(0);
        if (outputDim != upW.size(0)) {
            return null;
        }
        long[] outputShape = new long[] { input.size(0), input.size(1), outputDim };
        AccelTensor buffer = outputBuffer;
        if (buffer != null && !buffer.hasShape(outputShape)) {
            buffer = null;
        }

        long t0 = System.nanoTime();
        AccelTensor combined = AccelOps.fusedGatedHalfLinear(
                input,
                gateW,
                gateB,
                upW,
                upB,
                activationType == FFNActivationType.GELU,
                buffer);
        if (combined != null) {
            DirectInferenceProfiler.recordLinearNanos(PROFILE_KEY, System.nanoTime() - t0);
        }
        return combined;
    }

    private static boolean shouldPreferSeparateMetalHalfFfn(ModelConfigTraits traits,
                                                            boolean metalLinearEnabled,
                                                            AccelTensor input,
                                                            AccelTensor gateW,
                                                            AccelTensor upW) {
        return DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled,
                input,
                gateW,
                traits,
                GATE_UP_PAIR_PROFILE_KEY)
                && DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled,
                input,
                upW,
                traits,
                GATE_UP_PAIR_PROFILE_KEY);
    }
}
