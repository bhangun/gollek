/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;

final class DirectForwardGatedFfnFastPaths {
    private DirectForwardGatedFfnFastPaths() {
    }

    static AccelTensor tryComplete(DirectForwardGatedFfnRequest request) {
        boolean traceFfnFastPath = DirectForwardFfnFastPathTrace.isEnabled();
        if (traceFfnFastPath || DirectForwardMetalMatvecGatedFfn.shouldAttempt(
                request.input(), request.traits(), request.activationType())) {
            AccelTensor metalMatvecFfn = DirectForwardMetalMatvecGatedFfn.tryFfn(
                    request.runtime().log(),
                    request.runtime().metalBinding(),
                    request.runtime().capabilities(),
                    request.traits(),
                    request.config(),
                    request.metalLinearEnabled(),
                    request.decodeLogitsPhase(),
                    request.input(),
                    request.activationType(),
                    request.gateW(),
                    request.gateB(),
                    request.upW(),
                    request.upB(),
                    request.downW(),
                    request.downB(),
                    request.downOutputBuffer());
            if (metalMatvecFfn != null) {
                return metalMatvecFfn;
            }
        } else {
            DirectInferenceProfiler.recordFfnPath("matvec-gated-ffn:skip:attempt_gate");
        }

        boolean rowPrefillCandidate = traceFfnFastPath || DirectForwardMetalMatvecRowsGatedFfn.shouldAttempt(request);
        boolean fusedBeforeRows = !traceFfnFastPath
                && rowPrefillCandidate
                && DirectForwardMetalMatvecRowsGatedFfn.shouldDeferToFusedPrefill(request);
        if (fusedBeforeRows) {
            DirectForwardMetalMatvecRowsGatedFfn.recordStrategySkip(
                    request, "strategy_prefers_fused_geglu_prefill");
            AccelTensor metalFused = tryFused(request, traceFfnFastPath);
            if (metalFused != null) {
                return metalFused;
            }
        }

        if (rowPrefillCandidate) {
            AccelTensor metalMatvecRowsFfn = DirectForwardMetalMatvecRowsGatedFfn.tryFfn(request);
            if (metalMatvecRowsFfn != null) {
                return metalMatvecRowsFfn;
            }
        }

        if (fusedBeforeRows) {
            return null;
        }

        return tryFused(request, traceFfnFastPath);
    }

    private static AccelTensor tryFused(DirectForwardGatedFfnRequest request, boolean traceFfnFastPath) {
        String fusedSkipReason = traceFfnFastPath
                ? null
                : DirectForwardMetalFusedGatedFfn.skipReason(
                        request.runtime().metalBinding(),
                        request.runtime().capabilities(),
                        request.traits(),
                        request.metalLinearEnabled(),
                        request.input(),
                        request.activationType(),
                        request.gateB(),
                        request.upB(),
                        request.downB());
        if (traceFfnFastPath || fusedSkipReason == null) {
            AccelTensor metalFused = DirectForwardMetalFusedGatedFfn.tryFfn(
                    request.runtime().log(),
                    request.runtime().metalBinding(),
                    request.runtime().capabilities(),
                    request.traits(),
                    request.config(),
                    request.metalLinearEnabled(),
                    request.decodeLogitsPhase(),
                    request.input(),
                    request.activationType(),
                    request.gateW(),
                    request.gateB(),
                    request.upW(),
                    request.upB(),
                    request.downW(),
                    request.downB(),
                    request.downOutputBuffer());
            if (metalFused != null) {
                return metalFused;
            }
        } else {
            DirectInferenceProfiler.recordFfnPath("fused-gated-ffn:skip:" + fusedSkipReason);
        }

        return null;
    }

    static AccelTensor tryCombined(DirectForwardGatedFfnRequest request, AccelTensor combinedBuffer) {
        AccelTensor fusedCombined = DirectForwardLocalFusedGatedFfn.tryFfn(
                request.traits(),
                request.metalLinearEnabled(),
                request.input(),
                request.activationType(),
                request.gateW(),
                request.gateB(),
                request.upW(),
                request.upB(),
                combinedBuffer);
        if (fusedCombined != null) {
            return fusedCombined;
        }

        return DirectForwardMetalGateUpMatvecFfn.tryFfn(
                request.runtime().log(),
                request.runtime().metalBinding(),
                request.runtime().capabilities(),
                request.traits(),
                request.config(),
                request.metalLinearEnabled(),
                request.decodeLogitsPhase(),
                request.input(),
                request.activationType(),
                request.gateW(),
                request.gateB(),
                request.upW(),
                request.upB(),
                combinedBuffer);
    }
}
