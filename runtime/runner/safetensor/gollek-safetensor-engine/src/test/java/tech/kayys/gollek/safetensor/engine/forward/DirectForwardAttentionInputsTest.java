/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.attention.AttentionInput;
import tech.kayys.gollek.safetensor.engine.generation.attention.SharedKvState;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardAttentionInputsTest {

    @Test
    void createsAttentionWeightBundleFromLayerWeightsWithoutReorderingTensors() {
        AccelTensor qW = AccelTensor.zeros(2, 2);
        AccelTensor kW = AccelTensor.zeros(2, 2);
        AccelTensor vW = AccelTensor.zeros(2, 2);
        AccelTensor oW = AccelTensor.zeros(2, 2);
        AccelTensor qB = AccelTensor.zeros(2);
        AccelTensor kB = AccelTensor.zeros(2);
        AccelTensor vB = AccelTensor.zeros(2);
        AccelTensor oB = AccelTensor.zeros(2);
        AccelTensor attentionNormW = AccelTensor.zeros(2);
        AccelTensor qNormW = AccelTensor.zeros(2);
        AccelTensor kNormW = AccelTensor.zeros(2);
        AccelTensor postAttnNormW = AccelTensor.zeros(2);

        try {
            DirectForwardAttentionWeights weights = DirectForwardAttentionWeights.fromLayer(layerWeights(
                    qW, kW, vW, oW,
                    qB, kB, vB, oB,
                    attentionNormW,
                    qNormW, kNormW, postAttnNormW));

            assertSame(qW, weights.queryWeight());
            assertSame(kW, weights.keyWeight());
            assertSame(vW, weights.valueWeight());
            assertSame(oW, weights.outputWeight());
            assertSame(qB, weights.queryBias());
            assertSame(kB, weights.keyBias());
            assertSame(vB, weights.valueBias());
            assertSame(oB, weights.outputBias());
            assertSame(attentionNormW, weights.attentionNormWeight());
            assertSame(qNormW, weights.queryNormWeight());
            assertSame(kNormW, weights.keyNormWeight());
            assertSame(postAttnNormW, weights.postAttnNormWeight());
        } finally {
            closeAll(qW, kW, vW, oW, qB, kB, vB, oB, attentionNormW, qNormW, kNormW, postAttnNormW);
        }
    }

    @Test
    void factoryBuildsCausalLayerInputFromStageContextAndBundle() {
        ModelConfig config = new ModelConfig();
        ForwardWorkspace workspace = new ForwardWorkspace();
        workspace.ensureCapacity(2, 2, 4);
        AccelTensor normalizedInput = AccelTensor.zeros(1, 1, 2);
        AccelTensor qW = AccelTensor.zeros(2, 2);
        AccelTensor kW = AccelTensor.zeros(2, 2);
        AccelTensor vW = AccelTensor.zeros(2, 2);
        AccelTensor oW = AccelTensor.zeros(2, 2);
        AccelTensor qB = AccelTensor.zeros(2);
        AccelTensor kB = AccelTensor.zeros(2);
        AccelTensor vB = AccelTensor.zeros(2);
        AccelTensor oB = AccelTensor.zeros(2);
        AccelTensor qNormW = AccelTensor.zeros(2);
        AccelTensor kNormW = AccelTensor.zeros(2);
        AccelTensor postAttnNormW = AccelTensor.zeros(2);
        Map<Integer, SharedKvState> sharedKvStates = Map.of();

        try {
            DirectForwardLayerStageContext ctx = new DirectForwardLayerStageContext(
                    null,
                    workspace.getHiddenASeg(),
                    workspace.getHiddenBSeg(),
                    config,
                    null,
                    7,
                    1,
                    new long[] { 1, 1, 2 },
                    workspace,
                    null,
                    false,
                    false,
                    false,
                    false);
            DirectForwardAttentionWeights weights = new DirectForwardAttentionWeights(
                    qW, kW, vW, oW,
                    qB, kB, vB, oB,
                    null,
                    qNormW, kNormW, postAttnNormW);

            AttentionInput input = DirectForwardAttentionInputs.causalLayerInput(
                    ctx,
                    weights,
                    normalizedInput,
                    null,
                    3,
                    sharedKvStates);

            assertSame(normalizedInput, input.x);
            assertSame(qW, input.qW);
            assertSame(kW, input.kW);
            assertSame(vW, input.vW);
            assertSame(oW, input.oW);
            assertSame(qB, input.qB);
            assertSame(kB, input.kB);
            assertSame(vB, input.vB);
            assertSame(oB, input.oB);
            assertSame(config, input.config);
            assertEquals(7, input.layerIdx);
            assertEquals(3, input.startPos);
            assertSame(qNormW, input.qNormW);
            assertSame(kNormW, input.kNormW);
            assertSame(postAttnNormW, input.postAttnNormW);
            assertSame(sharedKvStates, input.sharedKvStates);
            assertSame(workspace.getHiddenBSeg(), input.attentionContextBuffer);
            assertSame(workspace.getNormedFfnSeg(), input.attentionOutputBuffer);
            assertTrue(input.isCausal);
        } finally {
            closeAll(normalizedInput, qW, kW, vW, oW, qB, kB, vB, oB, qNormW, kNormW, postAttnNormW);
            workspace.close();
        }
    }

    private static ResolvedLayerWeights layerWeights(
            AccelTensor qW,
            AccelTensor kW,
            AccelTensor vW,
            AccelTensor oW,
            AccelTensor qB,
            AccelTensor kB,
            AccelTensor vB,
            AccelTensor oB,
            AccelTensor attentionNormW,
            AccelTensor qNormW,
            AccelTensor kNormW,
            AccelTensor postAttnNormW) {
        return new ResolvedLayerWeights(
                qW, kW, vW, oW,
                qB, kB, vB, oB,
                attentionNormW,
                null, null,
                null, null,
                null, null,
                null, null,
                qNormW,
                kNormW,
                postAttnNormW,
                null, null, null, null,
                0.0f);
    }

    private static void closeAll(AccelTensor... tensors) {
        for (AccelTensor tensor : tensors) {
            tensor.close();
        }
    }
}
