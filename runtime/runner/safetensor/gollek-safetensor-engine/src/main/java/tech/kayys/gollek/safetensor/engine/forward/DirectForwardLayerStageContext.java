/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;

record DirectForwardLayerStageContext(
        DirectForwardSequenceContext sequenceContext,
        MemorySegment hiddenIn,
        MemorySegment hiddenOut,
        ModelConfig config,
        ModelArchitecture arch,
        int layerIdx,
        int seqLen,
        long[] hiddenShape,
        ForwardWorkspace workspace,
        ResolvedLayerWeights layerWeights,
        boolean addOneRmsNorm,
        boolean verboseLayers,
        boolean useMetalElementwise,
        boolean useNativeElementwiseAdd) {

    DirectForwardRuntimeContext runtime() {
        return sequenceContext.runtime();
    }

    ModelConfigTraits traits() {
        return sequenceContext.traits();
    }

    DirectForwardOperators operators() {
        return sequenceContext.operators();
    }

    static DirectForwardLayerStageContext create(DirectForwardSequenceContext sequenceContext,
                                                 MemorySegment hiddenIn,
                                                 MemorySegment hiddenOut,
                                                 ModelConfig config,
                                                 ModelArchitecture arch,
                                                 int layerIdx,
                                                 int seqLen,
                                                 long[] hiddenShape,
                                                 ForwardWorkspace workspace,
                                                 ResolvedLayerWeights layerWeights,
                                                 ResolvedModelWeights resolvedWeights) {
        DirectForwardRuntimeContext runtime = sequenceContext.runtime();
        ModelConfigTraits traits = sequenceContext.traits();
        boolean useMetalElementwise = runtime.canUseMetalElementwise(traits, seqLen);
        boolean useNativeElementwiseAdd = useMetalElementwise
                && runtime.capabilities().nativeElementwiseKernelsAvailable();
        return new DirectForwardLayerStageContext(
                sequenceContext,
                hiddenIn,
                hiddenOut,
                config,
                arch,
                layerIdx,
                seqLen,
                hiddenShape,
                workspace,
                layerWeights,
                resolvedWeights.addOneRmsNorm(),
                DirectForwardExecutionOptions.verboseLayersEnabled(),
                useMetalElementwise,
                useNativeElementwiseAdd);
    }
}
