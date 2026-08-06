/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardPrefillRequestTest {

    @Test
    void tokenIdRequestDefersEmbeddingPreparationToSequenceRunner() {
        long[] inputIds = {11L, 29L};
        ModelConfig config = new ModelConfig();
        ResolvedModelWeights resolvedWeights = resolvedWeights();

        DirectForwardPrefillRequest request =
                DirectForwardPrefillRequest.tokenIds(inputIds, Map.of(), config, null, null, resolvedWeights);

        assertNull(request.embeddings());
        assertNull(request.perLayerInputs());
        assertSame(inputIds, request.inputIds());
        assertSame(config, request.config());
        assertSame(resolvedWeights, request.resolvedWeights());
        assertFalse(request.embeddingsAlreadyInWorkspace());
    }

    @Test
    void preparedTokenPrefillBecomesWorkspaceBackedEmbeddingRequest() {
        long[] inputIds = {5L, 8L};
        AccelTensor embeddings = AccelTensor.zeros(1, 2, 3);
        AccelTensor[] perLayerInputs = {AccelTensor.zeros(1, 2, 3)};
        DirectForwardInputPreparation.PreparedTokenPrefill prepared =
                new DirectForwardInputPreparation.PreparedTokenPrefill(embeddings, perLayerInputs);
        DirectForwardPrefillRequest request =
                DirectForwardPrefillRequest.tokenIds(inputIds, Map.of(), new ModelConfig(), null, null,
                        resolvedWeights());

        try {
            DirectForwardPrefillRequest preparedRequest = request.withPreparedTokenPrefill(prepared);

            assertSame(embeddings, preparedRequest.embeddings());
            assertSame(perLayerInputs, preparedRequest.perLayerInputs());
            assertArrayEquals(inputIds, preparedRequest.inputIds());
            assertTrue(preparedRequest.embeddingsAlreadyInWorkspace());
        } finally {
            prepared.close();
        }
    }

    private static ResolvedModelWeights resolvedWeights() {
        return new ResolvedModelWeights(
                "test",
                "test",
                0,
                false,
                null,
                null,
                null,
                null,
                null,
                null,
                new ResolvedLayerWeights[0]);
    }
}
