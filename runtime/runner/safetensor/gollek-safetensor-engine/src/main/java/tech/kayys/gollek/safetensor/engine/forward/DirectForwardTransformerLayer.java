/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.attention.SharedKvState;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;
import java.util.Map;

final class DirectForwardTransformerLayer {
    private DirectForwardTransformerLayer() {
    }

    static void forward(DirectForwardSequenceContext context,
                        MemorySegment hiddenIn,
                        MemorySegment hiddenOut,
                        AccelTensor perLayerInput,
                        Map<String, AccelTensor> weights,
                        ModelConfig config,
                        ModelArchitecture arch,
                        KVCacheManager.KVCacheSession kvCache,
                        int layerIdx,
                        int startPos,
                        int seqLen,
                        long[] hiddenShape,
                        ForwardWorkspace ws,
                        Map<Integer, SharedKvState> sharedKvStates,
                        ResolvedLayerWeights layerWeights,
                        ResolvedModelWeights resolvedWeights) {
        DirectForwardLayerStageContext stageContext = DirectForwardLayerStageContext.create(
                context,
                hiddenIn,
                hiddenOut,
                config,
                arch,
                layerIdx,
                seqLen,
                hiddenShape,
                ws,
                layerWeights,
                resolvedWeights);

        if (stageContext.verboseLayers()) {
            System.err.println("  -> AttentionStage");
        }
        DirectForwardAttentionStage.run(stageContext, context.attentionKernel(), kvCache, startPos, sharedKvStates);
        if (stageContext.verboseLayers()) {
            System.err.println("  -> FfnStage");
        }
        DirectForwardFfnStage.run(stageContext, context.moeForwardPass(), weights);
        if (stageContext.verboseLayers()) {
            System.err.println("  -> LayerTailStage");
        }
        DirectForwardLayerTailStage.run(stageContext, perLayerInput);
        if (stageContext.verboseLayers()) {
            System.err.println("  -> Layer complete");
        }
    }

}
