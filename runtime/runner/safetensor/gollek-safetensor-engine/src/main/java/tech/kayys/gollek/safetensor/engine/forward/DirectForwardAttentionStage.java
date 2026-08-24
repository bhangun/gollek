/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardDiagnostics.logSegmentStats;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardDiagnostics.logTensorStats;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.safetensor.engine.generation.attention.AttentionInput;
import tech.kayys.gollek.safetensor.engine.generation.attention.FlashAttentionKernel;
import tech.kayys.gollek.safetensor.engine.generation.attention.SharedKvState;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;

import java.lang.foreign.MemorySegment;
import java.util.Map;

final class DirectForwardAttentionStage {
    private DirectForwardAttentionStage() {
    }

    static void run(DirectForwardLayerStageContext ctx,
                    FlashAttentionKernel attentionKernel,
                    KVCacheManager.KVCacheSession kvCache,
                    int startPos,
                    Map<Integer, SharedKvState> sharedKvStates) {
        DirectForwardAttentionWeights attentionWeights = DirectForwardAttentionWeights.fromLayer(ctx.layerWeights());
        MemorySegment normedAttnSeg = ctx.workspace().getNormedAttnSeg();
        DirectForwardNorms.rmsNormRows(normedAttnSeg, ctx.hiddenIn(),
                attentionWeights.attentionNormWeight(), ctx.hiddenShape(),
                ctx.seqLen(), ctx.config(), ctx.addOneRmsNorm(), ctx.useMetalElementwise(), ctx.runtime());

        AttentionInput attnIn = DirectForwardAttentionInputs.causalLayerInput(
                ctx,
                attentionWeights,
                AccelTensor.view(normedAttnSeg, ctx.hiddenShape()),
                kvCache,
                startPos,
                sharedKvStates);

        // Qwen3.6 attn_output_gate hack: slice q_proj weight in half
        // so standard FlashAttention gets the correct head dimension.
        if (attnIn.qW != null) {
            long expectedQRows = (long) ctx.config().numAttentionHeads() * ctx.config().resolvedHeadDim();
            if (expectedQRows > 0 && attnIn.qW.size(0) == expectedQRows * 2) {
                if (ctx.verboseLayers()) {
                    System.err.printf("[DEBUG] Layer %d slicing qW from %d to %d for attn_output_gate%n",
                            ctx.layerIdx(), attnIn.qW.size(0), expectedQRows);
                }
                AccelTensor slicedQw = attnIn.qW.slice(0, 0, expectedQRows).squeeze();
                AccelTensor slicedQb = attnIn.qB != null ? attnIn.qB.slice(0, 0, expectedQRows).squeeze() : null;
                attnIn = new AttentionInput(attnIn.x, slicedQw, attnIn.kW, attnIn.vW, attnIn.oW,
                        slicedQb, attnIn.kB, attnIn.vB, attnIn.oB,
                        attnIn.arch, attnIn.config, attnIn.kvCache,
                        attnIn.layerIdx, attnIn.startPos, attnIn.isCausal,
                        attnIn.qNormW, attnIn.kNormW, attnIn.postAttnNormW,
                        attnIn.sharedKvStates, attnIn.attentionContextBuffer,
                        attnIn.attentionOutputBuffer);
            }
        }

        if (attnIn.qW == null && !ctx.arch().hasFusedQKV()) {
            // Unsupported attention block (e.g. Mamba linear_attn hybrid). 
            // Skip the attention block by passing the hidden state directly.
            if (ctx.verboseLayers()) {
                System.err.printf("[DEBUG] Layer %d Attention skipped (unsupported block type)%n", ctx.layerIdx());
            }
            MemorySegment.copy(ctx.hiddenIn(), 0, ctx.hiddenOut(), 0, ctx.hiddenIn().byteSize());
            return;
        }

        if (ctx.verboseLayers()) {
            System.err.printf("[DEBUG] Layer %d Attention start%n", ctx.layerIdx());
            System.err.flush();
        }
        long tAttention0 = System.nanoTime();
        AccelTensor attnOut = attentionKernel.compute(attnIn);
        DirectInferenceProfiler.recordAttentionNanos(System.nanoTime() - tAttention0);
        if (ctx.verboseLayers()) {
            logTensorStats(attnOut, "layer " + ctx.layerIdx() + " attnOut");
        }




        DirectForwardElementwiseOps.residualAdd(ctx.runtime().log(), ctx.runtime().metalBinding(),
                ctx.hiddenIn(), attnOut, ctx.hiddenOut(),
                ctx.seqLen(), ctx.config().hiddenSize(), ctx.useNativeElementwiseAdd());
        attnOut.close();
        if (ctx.verboseLayers()) {
            logSegmentStats(ctx.hiddenOut(), ctx.hiddenShape(), "layer " + ctx.layerIdx() + " postAttnResidual");
        }
    }
}
