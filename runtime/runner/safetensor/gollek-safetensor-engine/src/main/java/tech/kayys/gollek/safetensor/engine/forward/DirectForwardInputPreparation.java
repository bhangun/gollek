/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

final class DirectForwardInputPreparation {
    private DirectForwardInputPreparation() {
    }

    record PreparationContext(
            ModelConfigTraits traits,
            DirectForwardOperators operators) {
    }

    record TokenPrefillRequest(
            long[] inputIds,
            ModelConfig config,
            ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache,
            ResolvedModelWeights resolvedWeights) {
    }

    record DecodeTokenRequest(
            long tokenId,
            ModelConfig config,
            ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache,
            ResolvedModelWeights resolvedWeights) {
    }

    record PreparedTokenPrefill(
            AccelTensor embeddings,
            AccelTensor[] perLayerInputs) implements AutoCloseable {
        @Override
        public void close() {
            DirectForwardPerLayerInputs.close(perLayerInputs);
            embeddings.closeWithParent();
        }
    }

    record PreparedDecodeToken(
            long[] hiddenShape,
            AccelTensor[] perLayerInputs,
            AccelTensor hiddenView) implements AutoCloseable {
        @Override
        public void close() {
            DirectForwardPerLayerInputs.close(perLayerInputs);
            if (hiddenView != null) {
                hiddenView.close();
            }
        }
    }

    static PreparedTokenPrefill prepareTokenPrefill(
            PreparationContext context,
            TokenPrefillRequest request) {
        ResolvedModelWeights resolvedWeights = request.resolvedWeights();
        AccelTensor embedTable = resolvedWeights.embedTokens();
        if (embedTable == null) {
            throw new IllegalStateException("Missing embed tokens weight: " + request.arch().embedTokensWeight());
        }
        int seqLen = request.inputIds().length;
        if (seqLen < 1) {
            throw new IllegalArgumentException("Prompt must result in at least one token.");
        }

        ModelConfig config = request.config();
        ForwardWorkspace ws = request.kvCache().getWorkspace();
        ws.ensureCapacity((long) seqLen * config.hiddenSize(), config.hiddenSize(), config.intermediateSize());
        long[] hiddenShape = new long[] { 1L, seqLen, config.hiddenSize() };
        float scale = request.arch().embeddingScaleFactor((int) embedTable.size(-1));
        embedTable.copyRowsToFloatSegment(request.inputIds(), ws.getHiddenASeg(), scale);

        AccelTensor hidden = AccelTensor.view(ws.getHiddenASeg(), hiddenShape);
        AccelTensor[] perLayerInputs = null;
        try {
            perLayerInputs = DirectForwardPerLayerInputs.build(
                    request.inputIds(), hidden, context.traits(), config, resolvedWeights, context.operators());
            return new PreparedTokenPrefill(hidden, perLayerInputs);
        } catch (RuntimeException | Error e) {
            DirectForwardPerLayerInputs.close(perLayerInputs);
            hidden.closeWithParent();
            throw e;
        }
    }

    static PreparedDecodeToken prepareDecodeToken(
            PreparationContext context,
            DecodeTokenRequest request) {
        ResolvedModelWeights resolvedWeights = request.resolvedWeights();
        AccelTensor embedTable = resolvedWeights.embedTokens();
        if (embedTable == null) {
            throw new IllegalStateException("Missing embed tokens weight.");
        }

        ModelConfig config = request.config();
        ForwardWorkspace ws = request.kvCache().getWorkspace();
        ws.ensureCapacity(config.hiddenSize(), config.hiddenSize(), config.intermediateSize());
        long[] tokenHiddenShape = new long[] { 1L, 1L, config.hiddenSize() };
        float scale = request.arch().embeddingScaleFactor((int) embedTable.size(-1));
        embedTable.copyRowToFloatSegment(request.tokenId(), ws.getHiddenASeg(), scale);

        AccelTensor hiddenView = null;
        AccelTensor[] perLayerInputs = null;
        try {
            if (DirectForwardPerLayerInputs.needed(context.traits(), config, resolvedWeights)) {
                long[] tokenIds = { request.tokenId() };
                hiddenView = AccelTensor.view(ws.getHiddenASeg(), tokenHiddenShape);
                perLayerInputs = DirectForwardPerLayerInputs.build(
                        tokenIds, hiddenView, context.traits(), config, resolvedWeights, context.operators());
            }
            return new PreparedDecodeToken(tokenHiddenShape, perLayerInputs, hiddenView);
        } catch (RuntimeException | Error e) {
            DirectForwardPerLayerInputs.close(perLayerInputs);
            if (hiddenView != null) {
                hiddenView.close();
            }
            throw e;
        }
    }
}
