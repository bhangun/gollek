/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.jboss.logging.Logger;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardPerLayerResidualRequestTest {

    @Test
    void createsResidualWeightsFromLayerWithoutReorderingTensors() {
        AccelTensor gateWeight = AccelTensor.zeros(2, 2);
        AccelTensor projectionWeight = AccelTensor.zeros(2, 2);
        AccelTensor normWeight = AccelTensor.zeros(2);

        try {
            DirectForwardPerLayerResidualWeights weights =
                    DirectForwardPerLayerResidualWeights.fromLayer(layerWeights(
                            gateWeight,
                            projectionWeight,
                            normWeight));

            assertSame(gateWeight, weights.gateWeight());
            assertSame(projectionWeight, weights.projectionWeight());
            assertSame(normWeight, weights.normWeight());
            assertTrue(weights.complete());
        } finally {
            gateWeight.close();
            projectionWeight.close();
            normWeight.close();
        }
    }

    @Test
    void incompleteResidualWeightsAreRejected() {
        DirectForwardPerLayerResidualWeights weights =
                new DirectForwardPerLayerResidualWeights(null, AccelTensor.zeros(2, 2), AccelTensor.zeros(2));
        try {
            assertFalse(weights.complete());
        } finally {
            weights.projectionWeight().close();
            weights.normWeight().close();
        }
    }

    @Test
    void createsResidualRequestFromLayerStageContext() {
        ModelConfig config = new ModelConfig();
        DirectForwardRuntimeContext runtime = runtime();
        DirectForwardOperators operators = new DirectForwardOperators(runtime, ignored -> ModelConfigTraits.EMPTY);
        DirectForwardSequenceContext sequenceContext =
                new DirectForwardSequenceContext(runtime, null, null, ModelConfigTraits.EMPTY, operators);
        ForwardWorkspace workspace = new ForwardWorkspace();
        workspace.ensureCapacity(2, 2, 4);
        AccelTensor gateWeight = AccelTensor.zeros(2, 2);
        AccelTensor projectionWeight = AccelTensor.zeros(2, 2);
        AccelTensor normWeight = AccelTensor.zeros(2);
        AccelTensor perLayerInput = AccelTensor.zeros(1, 1, 2);
        long[] hiddenShape = { 1L, 1L, 2L };
        ResolvedLayerWeights layerWeights = layerWeights(gateWeight, projectionWeight, normWeight);

        try {
            DirectForwardLayerStageContext ctx = new DirectForwardLayerStageContext(
                    sequenceContext,
                    workspace.getHiddenASeg(),
                    workspace.getHiddenBSeg(),
                    config,
                    null,
                    3,
                    1,
                    hiddenShape,
                    workspace,
                    layerWeights,
                    true,
                    false,
                    true,
                    true);

            DirectForwardPerLayerResidualRequest request =
                    DirectForwardPerLayerResidualRequest.fromStage(ctx, perLayerInput);

            assertSame(workspace.getHiddenBSeg(), request.hiddenSeg());
            assertSame(hiddenShape, request.hiddenShape());
            assertSame(config, request.config());
            assertSame(workspace, request.workspace());
            assertSame(perLayerInput, request.perLayerInput());
            assertSame(runtime, request.runtime());
            assertSame(operators, request.operators());
            assertSame(gateWeight, request.weights().gateWeight());
            assertSame(projectionWeight, request.weights().projectionWeight());
            assertSame(normWeight, request.weights().normWeight());
        } finally {
            gateWeight.close();
            projectionWeight.close();
            normWeight.close();
            perLayerInput.close();
            workspace.close();
        }
    }

    private static ResolvedLayerWeights layerWeights(
            AccelTensor gateWeight,
            AccelTensor projectionWeight,
            AccelTensor normWeight) {
        return new ResolvedLayerWeights(
                null, null, null, null,
                null, null, null, null,
                null,
                null, null,
                null, null,
                null, null,
                null, null,
                null, null, null,
                gateWeight,
                projectionWeight,
                normWeight,
                null,
                0.0f);
    }

    private static DirectForwardRuntimeContext runtime() {
        return new DirectForwardRuntimeContext(
                Logger.getLogger(DirectForwardPerLayerResidualRequestTest.class),
                null,
                DirectForwardMetalCapabilities.EMPTY,
                false,
                false);
    }
}
