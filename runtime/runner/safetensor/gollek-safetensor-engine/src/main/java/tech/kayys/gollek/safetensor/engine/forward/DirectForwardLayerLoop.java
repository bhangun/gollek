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

final class DirectForwardLayerLoop {
    private DirectForwardLayerLoop() {
    }

    record Request(
            DirectForwardSequenceContext context,
            MemorySegment currentHidden,
            MemorySegment nextHidden,
            AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights,
            ModelConfig config,
            ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache,
            int startPos,
            int seqLen,
            long[] hiddenShape,
            ForwardWorkspace workspace,
            Map<Integer, SharedKvState> sharedKvStates,
            ResolvedModelWeights resolvedWeights,
            boolean verboseTokens,
            String debugLabel) {
    }

    static MemorySegment run(Request request) {
        MemorySegment currentHidden = request.currentHidden();
        MemorySegment nextHidden = request.nextHidden();
        for (int layer = 0; layer < request.config().getNumHiddenLayers(); layer++) {
            logLayerStart(request, layer);
            DirectForwardTransformerLayer.forward(
                    request.context(),
                    currentHidden,
                    nextHidden,
                    perLayerInput(request, layer),
                    request.weights(),
                    request.config(),
                    request.arch(),
                    request.kvCache(),
                    layer,
                    request.startPos(),
                    request.seqLen(),
                    request.hiddenShape(),
                    request.workspace(),
                    request.sharedKvStates(),
                    request.resolvedWeights().layer(layer),
                    request.resolvedWeights());

            MemorySegment temp = currentHidden;
            currentHidden = nextHidden;
            nextHidden = temp;
        }
        return currentHidden;
    }

    private static void logLayerStart(Request request, int layer) {
        if (!request.verboseTokens() || request.debugLabel() == null) {
            return;
        }
        System.err.printf("[DEBUG] %s %d/%d start%n",
                request.debugLabel(), layer, request.config().getNumHiddenLayers());
        System.err.flush();
    }

    private static AccelTensor perLayerInput(Request request, int layer) {
        return request.perLayerInputs() == null ? null : request.perLayerInputs()[layer];
    }
}
