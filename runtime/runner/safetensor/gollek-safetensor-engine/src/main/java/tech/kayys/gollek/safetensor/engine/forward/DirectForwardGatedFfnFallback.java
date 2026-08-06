/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.elementCount;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.reusableWorkspaceView;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;

final class DirectForwardGatedFfnFallback {
    private DirectForwardGatedFfnFallback() {
    }

    static AccelTensor forward(DirectForwardGatedFfnRequest request) {
        long x0 = request.input().size(0);
        long x1 = request.input().size(1);
        long xLast = request.input().size(-1);
        long gateOutDim = request.gateW() != null && request.gateW().rank() > 0 ? request.gateW().size(0) : 0L;
        long[] gateShape = new long[] { x0, x1, gateOutDim };
        long gateElementCount = elementCount(gateShape);
        AccelTensor combinedBuffer = combinedWorkspace(request, xLast, gateOutDim, gateElementCount, gateShape);

        boolean localCombined = false;
        if (combinedBuffer == null && request.gateW() != null && request.upW() != null && gateOutDim == request.upW().size(0)) {
            combinedBuffer = AccelTensor.zeros(gateShape);
            localCombined = true;
        }

        AccelTensor acceleratedCombined = DirectForwardGatedFfnFastPaths.tryCombined(request, combinedBuffer);
        if (acceleratedCombined != null) {
            return downProjectAndClose(request, acceleratedCombined, localCombined ? combinedBuffer : null);
        }

        if (localCombined && combinedBuffer != null) {
            combinedBuffer.close();
        }

        AccelTensor gateBuffer = null;
        AccelTensor upBuffer = null;
        boolean reuseFfnProjectionWorkspace = DirectForwardExecutionOptions.canReuseFfnProjectionWorkspace();
        if (reuseFfnProjectionWorkspace && request.ws() != null) {
            request.ws().ensureProjectionScratchCapacity(gateElementCount * Float.BYTES);
            gateBuffer = reusableWorkspaceView(request.ws().getGateSeg(), gateShape);
            upBuffer = reusableWorkspaceView(request.ws().getUpSeg(), gateShape);
        }

        DirectForwardMetalHalfLinearPair.Result pairedGateUp = DirectForwardMetalHalfLinearPair.tryPair(
                request.runtime().log(),
                request.runtime().metalBinding(),
                request.runtime().capabilities(),
                request.traits(),
                request.config(),
                request.metalLinearEnabled(),
                request.decodeLogitsPhase(),
                request.input(),
                request.gateW(),
                request.gateB(),
                request.upW(),
                request.upB(),
                gateBuffer,
                upBuffer);
        AccelTensor gate;
        AccelTensor up;
        if (pairedGateUp != null) {
            gate = pairedGateUp.first();
            up = pairedGateUp.second();
        } else {
            if (reuseFfnProjectionWorkspace && request.ws() != null) {
                gateBuffer = reusableWorkspaceView(request.ws().getGateSeg(), gateShape);
                upBuffer = reusableWorkspaceView(request.ws().getUpSeg(), gateShape);
            }
            gate = linear(request, request.input(), request.gateW(), request.gateB(), "ffn_gate", gateBuffer);
            up = linear(request, request.input(), request.upW(), request.upB(), "ffn_up", upBuffer);
        }

        AccelTensor combined = activate(request, gate, up, gateShape, Math.toIntExact(gateElementCount));
        if (!gate.isClosed()) {
            gate.close();
        }
        up.close();
        return downProjectAndClose(request, combined, localCombined ? combinedBuffer : null);
    }

    private static AccelTensor combinedWorkspace(DirectForwardGatedFfnRequest request,
                                                 long inputLastDim,
                                                 long gateOutDim,
                                                 long gateElementCount,
                                                 long[] gateShape) {
        ForwardWorkspace ws = request.ws();
        if (ws == null || request.gateW() == null || request.upW() == null
                || request.gateW().rank() != 2
                || request.upW().rank() != 2
                || gateOutDim != request.upW().size(0)) {
            return null;
        }
        ws.ensureCapacity(request.input().numel(), inputLastDim, gateOutDim);
        long requiredCombinedBytes = gateElementCount * Float.BYTES;
        if (ws.getCombinedSeg() != null && ws.getCombinedSeg().byteSize() >= requiredCombinedBytes) {
            return AccelTensor.view(ws.getCombinedSeg(), gateShape);
        }
        return null;
    }

    private static AccelTensor activate(DirectForwardGatedFfnRequest request,
                                        AccelTensor gate,
                                        AccelTensor up,
                                        long[] gateShape,
                                        int gateElements) {
        AccelTensor activationWorkspace = request.ws() != null
                ? AccelTensor.view(request.ws().getCombinedSeg(), gateShape)
                : null;
        return DirectForwardGatedActivation.combine(
                request.runtime().metalBinding(),
                request.activationType(),
                request.runtime().canUseMetalElementwise(request.traits(), Math.toIntExact(request.input().size(1))),
                gate,
                up,
                activationWorkspace,
                gateElements);
    }

    private static AccelTensor downProjectAndClose(DirectForwardGatedFfnRequest request,
                                                   AccelTensor combined,
                                                   AccelTensor localCombinedBuffer) {
        AccelTensor out = DirectForwardLinearProjection.ffnDownLinear(linearRequest(
                request,
                combined,
                request.downW(),
                request.downB(),
                "ffn_down",
                request.downOutputBuffer()));
        if (request.ws() == null || combined.dataPtr() != request.ws().getCombinedSeg()) {
            combined.close();
        }
        if (localCombinedBuffer != null && !localCombinedBuffer.isClosed() && localCombinedBuffer != combined) {
            localCombinedBuffer.close();
        }
        return out;
    }

    private static AccelTensor linear(DirectForwardGatedFfnRequest request,
                                      AccelTensor input,
                                      AccelTensor weight,
                                      AccelTensor bias,
                                      String profileKey,
                                      AccelTensor outputBuffer) {
        return DirectForwardLinearProjection.linear(linearRequest(
                request,
                input,
                weight,
                bias,
                profileKey,
                outputBuffer));
    }

    private static DirectForwardLinearRequest linearRequest(DirectForwardGatedFfnRequest request,
                                                            AccelTensor input,
                                                            AccelTensor weight,
                                                            AccelTensor bias,
                                                            String profileKey,
                                                            AccelTensor outputBuffer) {
        return DirectForwardLinearRequest.gatedFfnProjection(
                request,
                input,
                weight,
                bias,
                profileKey,
                outputBuffer);
    }
}
