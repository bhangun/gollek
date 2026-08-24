/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardDiagnostics.logSegmentStats;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardDiagnostics.logTensorStats;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelOps;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.safetensor.engine.generation.moe.MoeForwardPass;

import java.lang.foreign.MemorySegment;
import java.util.Map;

final class DirectForwardFfnStage {
    private DirectForwardFfnStage() {
    }

    static void run(DirectForwardLayerStageContext ctx,
                    MoeForwardPass moeForwardPass,
                    Map<String, AccelTensor> weights) {
        ResolvedLayerWeights layerWeights = ctx.layerWeights();
        AccelTensor preFfnNormW = layerWeights.preFfnNormWeight();
        AccelTensor postFfnNormW = layerWeights.postFfnNormWeight();

        MemorySegment normedFfnSeg = ctx.workspace().getNormedFfnSeg();
        DirectForwardNorms.rmsNormRows(normedFfnSeg, ctx.hiddenOut(), preFfnNormW, ctx.hiddenShape(),
                ctx.seqLen(), ctx.config(), ctx.addOneRmsNorm(), ctx.useMetalElementwise(), ctx.runtime());

        long tFfn0 = System.nanoTime();
        AccelTensor mlpOutputBuffer = AccelTensor.view(ctx.hiddenIn(), ctx.hiddenShape());
        AccelTensor mlpIn = AccelTensor.view(normedFfnSeg, ctx.hiddenShape());
        AccelTensor mlpOut = ctx.config().isMoeLayer(ctx.layerIdx())
                ? moeForwardPass.computeAccel(mlpIn, weights, ctx.config(), ctx.arch(), ctx.layerIdx())
                : ctx.operators().swigluFfn(mlpIn, ctx.arch(), ctx.config(),
                        DirectForwardGatedFfnWeights.fromLayer(layerWeights),
                        ctx.workspace(), mlpOutputBuffer);
        DirectInferenceProfiler.recordFfnNanos(System.nanoTime() - tFfn0);
        if (ctx.verboseLayers()) {
            logTensorStats(mlpOut, "layer " + ctx.layerIdx() + " mlpOut");
        }

        AccelTensor mlpNormed;
        if (postFfnNormW != null) {
            if (ctx.useMetalElementwise() && DirectForwardElementwisePolicy.shouldUseMetalPostFfnNorm(ctx.traits())) {
                DirectForwardElementwiseOps.rmsNormRowsMetal(ctx.runtime().metalBinding(), normedFfnSeg,
                        mlpOut.dataPtr(), postFfnNormW.dataPtr(), ctx.seqLen(), ctx.config().hiddenSize(),
                        (float) ctx.config().rmsNormEps(), ctx.addOneRmsNorm());
                mlpNormed = AccelTensor.view(normedFfnSeg, ctx.hiddenShape());
            } else {
                mlpNormed = AccelOps.rmsNorm(mlpOut, postFfnNormW, ctx.config().rmsNormEps(),
                        ctx.addOneRmsNorm());
            }
            mlpOut.close();
        } else {
            mlpNormed = mlpOut;
        }


        DirectForwardElementwiseOps.residualAdd(ctx.runtime().log(), ctx.runtime().metalBinding(),
                ctx.hiddenOut(), mlpNormed, ctx.hiddenOut(),
                ctx.seqLen(), ctx.config().hiddenSize(), ctx.useNativeElementwiseAdd());
        mlpNormed.close();
        if (ctx.verboseLayers()) {
            logSegmentStats(ctx.hiddenOut(), ctx.hiddenShape(), "layer " + ctx.layerIdx() + " postFfnResidual");
        }
    }
}
