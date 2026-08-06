/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.selectLastToken;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelOps;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

final class DirectForwardOutputProjection {
    private DirectForwardOutputProjection() {
    }

    static AccelTensor prefillLogits(DirectForwardPrefillLogitsRequest request) {
        DirectForwardOutputProjectionContext context = request.context();
        AccelTensor hidden = AccelTensor.view(request.hiddenSegment(), request.hiddenShape());
        AccelTensor normed = finalNorm(context, hidden, request.hiddenShape(), request.seqLen());
        if (hidden != request.embeddings() && hidden != normed) {
            hidden.close();
        }

        AccelTensor lmHeadW = lmHeadOrThrow(context.resolvedWeights());
        if (request.verboseTokens() && request.seqLen() > 1) {
            DirectForwardLogits.debugSequencePositionLogits(normed, lmHeadW, request.seqLen(),
                    (input, weight, bias) -> linear(context,
                            false, input, weight, bias, null, null));
        }

        AccelTensor lastPos = selectLastToken(normed, request.seqLen());
        AccelTensor logitsOutput = DirectForwardLogits.reusableOutputTensor(context.workspace(), lastPos, lmHeadW);
        try {
            return projectLogits(context, false, lastPos, lmHeadW, logitsOutput);
        } finally {
            lastPos.closeWithParent();
        }
    }

    static AccelTensor decodeLogits(DirectForwardDecodeLogitsRequest request) {
        DirectForwardOutputProjectionContext context = request.context();
        AccelTensor finalHidden = AccelTensor.view(request.hiddenSegment(), request.hiddenShape());
        AccelTensor normed = finalNorm(context, finalHidden, request.hiddenShape(), 1);
        if (finalHidden != normed) {
            finalHidden.close();
        }

        AccelTensor lmHeadW = lmHeadOrThrow(context.resolvedWeights());
        AccelTensor logitsOutput = request.reuseLogitsOutput()
                ? DirectForwardLogits.reusableOutputTensor(context.workspace(), normed, lmHeadW)
                : null;
        try {
            return projectLogits(context, true, normed, lmHeadW, logitsOutput);
        } finally {
            if (normed.dataPtr() != context.workspace().getNormedAttnSeg()) {
                normed.close();
            }
        }
    }

    private static AccelTensor finalNorm(DirectForwardOutputProjectionContext context,
                                         AccelTensor hidden,
                                         long[] hiddenShape,
                                         int rows) {
        ResolvedModelWeights resolvedWeights = context.resolvedWeights();
        ModelConfig config = context.config();
        if (context.runtime().canUseMetalElementwise(context.traits(), rows)) {
            AccelTensor normed = AccelTensor.view(context.workspace().getNormedAttnSeg(), hiddenShape);
            DirectForwardElementwiseOps.rmsNormRowsMetal(
                    context.runtime().metalBinding(),
                    normed.dataPtr(),
                    hidden.dataPtr(),
                    resolvedWeights.finalNorm().dataPtr(),
                    rows,
                    config.getHiddenSize(),
                    (float) config.getRmsNormEps(),
                    resolvedWeights.addOneRmsNorm());
            return normed;
        }
        return AccelOps.rmsNorm(hidden, resolvedWeights.finalNorm(), config.getRmsNormEps(),
                resolvedWeights.addOneRmsNorm());
    }

    private static AccelTensor projectLogits(DirectForwardOutputProjectionContext context,
                                             boolean decodeLogitsPhase,
                                             AccelTensor input,
                                             AccelTensor lmHeadW,
                                             AccelTensor outputBuffer) {
        long tLogits0 = System.nanoTime();
        AccelTensor logits = linear(context, decodeLogitsPhase, input, lmHeadW, null, "logits", outputBuffer);
        
        if (context.config().getFinalLogitSoftcapping() != null) {
            float softcap = context.config().getFinalLogitSoftcapping().floatValue();
            if (softcap > 0.0f) {
                applyFinalSoftcap(logits, softcap);
            }
        }
        
        DirectInferenceProfiler.recordLogitsProjectionNanos(System.nanoTime() - tLogits0);
        return logits;
    }

    private static AccelTensor applyFinalSoftcap(AccelTensor logits, float softcap) {
        float[] floats = logits.toFloatArray();
        for (int i = 0; i < floats.length; i++) {
            floats[i] = (float) (Math.tanh(floats[i] / softcap) * softcap);
        }
        MemorySegment.copy(floats, 0, logits.dataPtr(), ValueLayout.JAVA_FLOAT, 0L, floats.length);
        return logits;
    }

    private static AccelTensor linear(DirectForwardOutputProjectionContext context,
                                      boolean decodeLogitsPhase,
                                      AccelTensor input,
                                      AccelTensor weight,
                                      AccelTensor bias,
                                      String profileKey,
                                      AccelTensor outputBuffer) {
        return DirectForwardLinearProjection.linear(DirectForwardLinearRequest.logitsProjection(
                new DirectForwardLinearContext(context.runtime(), context.traits(), context.config()),
                decodeLogitsPhase,
                input,
                weight,
                bias,
                profileKey,
                outputBuffer));
    }

    private static AccelTensor lmHeadOrThrow(ResolvedModelWeights resolvedWeights) {
        AccelTensor lmHeadW = resolvedWeights.lmHead();
        if (lmHeadW == null) {
            throw new IllegalStateException(
                    "Missing lm_head weight. Safetensor file might be incomplete or config.tie_word_embeddings is missing.");
        }
        return lmHeadW;
    }

}
