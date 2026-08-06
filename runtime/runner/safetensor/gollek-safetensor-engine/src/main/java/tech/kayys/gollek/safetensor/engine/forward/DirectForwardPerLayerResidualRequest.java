/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;

record DirectForwardPerLayerResidualRequest(
        MemorySegment hiddenSeg,
        long[] hiddenShape,
        int seqLen,
        ModelConfig config,
        ModelArchitecture arch,
        DirectForwardPerLayerResidualWeights weights,
        AccelTensor perLayerInput,
        ForwardWorkspace workspace,
        boolean useMetalElementwise,
        boolean useNativeElementwiseAdd,
        boolean addOneRmsNorm,
        DirectForwardRuntimeContext runtime,
        DirectForwardOperators operators) {

    static DirectForwardPerLayerResidualRequest fromStage(
            DirectForwardLayerStageContext ctx,
            AccelTensor perLayerInput) {
        return new DirectForwardPerLayerResidualRequest(
                ctx.hiddenOut(),
                ctx.hiddenShape(),
                ctx.seqLen(),
                ctx.config(),
                ctx.arch(),
                DirectForwardPerLayerResidualWeights.fromLayer(ctx.layerWeights()),
                perLayerInput,
                ctx.workspace(),
                ctx.useMetalElementwise(),
                ctx.useNativeElementwiseAdd(),
                ctx.addOneRmsNorm(),
                ctx.runtime(),
                ctx.operators());
    }
}
