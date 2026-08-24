/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;

final class DirectForwardSequenceRunner {
    private DirectForwardSequenceRunner() {
    }

    static AccelTensor prefillTokenIds(
            DirectForwardSequenceContext context,
            DirectForwardPrefillRequest request) {
        try (DirectForwardInputPreparation.PreparedTokenPrefill prepared =
                     DirectForwardInputPreparation.prepareTokenPrefill(
                             inputPreparationContext(context),
                             tokenPrefillRequest(request))) {
            return prefillEmbeddings(context, request.withPreparedTokenPrefill(prepared));
        }
    }

    static AccelTensor prefillEmbeddings(
            DirectForwardSequenceContext context,
            DirectForwardPrefillRequest request) {
        AccelTensor embeddings = request.embeddings();
        ModelConfig config = request.config();
        KVCacheManager.KVCacheSession kvCache = request.kvCache();
        ensureDirectFfnSupported(config);
        int startPos = kvCache.currentPos();
        long seqLen = embeddings.size(1);
        if (seqLen < 1) {
            throw new IllegalArgumentException(
                    "Invalid sequence length: " + seqLen + ". Prompt must result in at least one token.");
        }
        int seqLenInt = Math.toIntExact(seqLen);
        long[] hiddenShape = new long[] { 1L, seqLen, config.hiddenSize() };
        ForwardWorkspace ws = kvCache.getWorkspace();
        ws.ensureCapacity((long) seqLen * config.hiddenSize(), config.hiddenSize(), config.intermediateSize());

        if (!request.embeddingsAlreadyInWorkspace()) {
            MemorySegment.copy(embeddings.dataPtr(), 0, ws.getHiddenASeg(), 0,
                    (long) seqLen * config.hiddenSize() * Float.BYTES);
        }

        boolean verboseTokens = DirectForwardExecutionOptions.verboseTokensEnabled();
        MemorySegment finalHidden = DirectForwardLayerLoop.run(new DirectForwardLayerLoop.Request(
                context,
                ws.getHiddenASeg(),
                ws.getHiddenBSeg(),
                request.perLayerInputs(),
                request.weights(),
                config,
                request.arch(),
                kvCache,
                startPos,
                seqLenInt,
                hiddenShape,
                ws,
                DirectForwardSharedKvStates.forPrefill(config, kvCache, startPos),
                request.resolvedWeights(),
                verboseTokens,
                "Prefill Layer"));

        AccelTensor logits = DirectForwardOutputProjection.prefillLogits(new DirectForwardPrefillLogitsRequest(
                outputProjectionContext(context, config, request.resolvedWeights(), ws),
                finalHidden,
                hiddenShape,
                embeddings,
                seqLenInt,
                verboseTokens));
        kvCache.advance(seqLenInt);
        return logits;
    }

    static AccelTensor decodeToken(
            DirectForwardSequenceContext context,
            DirectForwardDecodeRequest request) {
        ModelConfig config = request.config();
        KVCacheManager.KVCacheSession kvCache = request.kvCache();
        ensureDirectFfnSupported(config);
        ForwardWorkspace ws = kvCache.getWorkspace();
        try (DirectForwardInputPreparation.PreparedDecodeToken prepared =
                     DirectForwardInputPreparation.prepareDecodeToken(
                             inputPreparationContext(context),
                             decodeTokenRequest(request))) {
            MemorySegment finalHidden = DirectForwardLayerLoop.run(new DirectForwardLayerLoop.Request(
                    context,
                    ws.getHiddenASeg(),
                    ws.getHiddenBSeg(),
                    prepared.perLayerInputs(),
                    request.weights(),
                    config,
                    request.arch(),
                    kvCache,
                    request.startPos(),
                    1,
                    prepared.hiddenShape(),
                    ws,
                    DirectForwardSharedKvStates.forDecode(config, kvCache),
                    request.resolvedWeights(),
                    false,
                    null));

            AccelTensor logits = DirectForwardOutputProjection.decodeLogits(new DirectForwardDecodeLogitsRequest(
                    outputProjectionContext(context, config, request.resolvedWeights(), ws),
                    finalHidden,
                    prepared.hiddenShape(),
                    request.reuseLogitsOutput()));
            kvCache.advance(1);
            return logits;
        }
    }

    private static DirectForwardInputPreparation.PreparationContext inputPreparationContext(
            DirectForwardSequenceContext context) {
        return new DirectForwardInputPreparation.PreparationContext(
                context.traits(),
                context.operators());
    }

    private static DirectForwardInputPreparation.TokenPrefillRequest tokenPrefillRequest(
            DirectForwardPrefillRequest request) {
        return new DirectForwardInputPreparation.TokenPrefillRequest(
                request.inputIds(),
                request.config(),
                request.arch(),
                request.kvCache(),
                request.resolvedWeights());
    }

    private static DirectForwardInputPreparation.DecodeTokenRequest decodeTokenRequest(
            DirectForwardDecodeRequest request) {
        return new DirectForwardInputPreparation.DecodeTokenRequest(
                request.tokenId(),
                request.config(),
                request.arch(),
                request.kvCache(),
                request.resolvedWeights());
    }

    private static DirectForwardOutputProjectionContext outputProjectionContext(
            DirectForwardSequenceContext context,
            ModelConfig config,
            ResolvedModelWeights resolvedWeights,
            ForwardWorkspace workspace) {
        return new DirectForwardOutputProjectionContext(
                context.runtime(),
                context.traits(),
                config,
                resolvedWeights,
                workspace);
    }

    private static void ensureDirectFfnSupported(ModelConfig config) {
        if (config != null && config.requiresGemma4PackedMoeRuntime()) {
            throw new UnsupportedOperationException(
                    "Gemma4 packed MoE FFN is not supported by the direct SafeTensor runtime yet. "
                            + "Use a GGUF/backend plugin with Gemma4 packed experts support, or load a dense Gemma4 text checkpoint.");
        }
    }
}
