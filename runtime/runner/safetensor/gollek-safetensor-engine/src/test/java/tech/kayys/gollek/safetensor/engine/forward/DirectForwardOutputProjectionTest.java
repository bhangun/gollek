/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.jboss.logging.Logger;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class DirectForwardOutputProjectionTest {

    @Test
    void prefillCanProjectWhenFinalNormIsIdentity() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "identity_norm_test",
                  "hidden_size": 3,
                  "intermediate_size": 4,
                  "vocab_size": 2
                }
                """, ModelConfig.class);
        ForwardWorkspace ws = new ForwardWorkspace();
        ws.ensureCapacity(6, 3, 4);
        MemorySegment hiddenSeg = ws.getHiddenASeg();
        for (int i = 0; i < 6; i++) {
            hiddenSeg.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 1.0f);
        }

        AccelTensor embeddings = AccelTensor.view(hiddenSeg, new long[] { 1, 2, 3 });
        AccelTensor lmHead = AccelTensor.zeros(2, 3);
        lmHead.setFlat(0, 1.0f);
        lmHead.setFlat(4, 1.0f);
        ResolvedModelWeights weights = new ResolvedModelWeights(
                "test",
                "identity_norm_test",
                0,
                false,
                null,
                null,
                null,
                null,
                null,
                lmHead,
                new ResolvedLayerWeights[0]);

        DirectForwardOutputProjectionContext projectionContext = new DirectForwardOutputProjectionContext(
                runtime(),
                ModelConfigTraits.create(config),
                config,
                weights,
                ws);
        AccelTensor logits = DirectForwardOutputProjection.prefillLogits(new DirectForwardPrefillLogitsRequest(
                projectionContext,
                hiddenSeg,
                new long[] { 1, 2, 3 },
                embeddings,
                2,
                false));
        try {
            assertArrayEquals(new float[] { 4.0f, 5.0f }, logits.toFloatArray(), 0.0001f);
        } finally {
            logits.close();
            embeddings.close();
            lmHead.close();
            ws.close();
        }
    }

    private static DirectForwardRuntimeContext runtime() {
        return new DirectForwardRuntimeContext(
                Logger.getLogger(DirectForwardOutputProjectionTest.class),
                null,
                DirectForwardMetalCapabilities.EMPTY,
                false,
                false);
    }
}
