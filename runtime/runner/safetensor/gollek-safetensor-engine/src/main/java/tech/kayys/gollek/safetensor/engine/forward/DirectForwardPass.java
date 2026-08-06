/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.attention.FlashAttentionKernel;
import tech.kayys.gollek.safetensor.engine.generation.moe.MoeForwardPass;

import java.util.Map;

/**
 * Full transformer forward pass using AccelTensor + Apple Accelerate.
 * No LibTorch dependency.
 */
@ApplicationScoped
public class DirectForwardPass {
    private static final Logger log = Logger.getLogger(DirectForwardPass.class);
    @Inject
    FlashAttentionKernel attentionKernel;
    @Inject
    MoeForwardPass moeForwardPass;
    @Inject
    DirectForwardRuntimeState runtimeState;
    @Inject
    DirectForwardModelContext modelContext;
    @Inject
    DirectForwardFfnService ffnService;
    
    public float[] prefill(long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor logits = prefillLogitsTensor(inputIds, weights, config, arch, kvCache);
        float[] result = DirectForwardLogits.materializeAndClose(logits);
        DirectForwardLogits.logPrefillDiagnostics(
                result,
                config,
                DirectForwardExecutionOptions.verboseTokensEnabled());
        return result;
    }

    public void clearResolvedModelWeights(Map<String, AccelTensor> weights) {
        modelContext.clearResolvedWeights(weights);
    }

    public float[] prefill(AccelTensor embeddings, long[] inputIds, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        return prefill(embeddings, inputIds, null, weights, config, arch, kvCache);
    }

    public float[] prefill(AccelTensor embeddings, long[] inputIds, AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache) {
        AccelTensor logits = prefillLogitsTensor(embeddings, inputIds, perLayerInputs, weights, config, arch, kvCache);
        return DirectForwardLogits.materializeAndClose(logits);
    }

    public AccelTensor prefillLogitsTensor(long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        ResolvedModelWeights resolvedWeights = modelContext.resolveWeights(weights, config, arch);
        DirectForwardRuntimeContext runtime = runtimeContext();
        return DirectForwardSequenceRunner.prefillTokenIds(
                sequenceContext(runtime, config, arch),
                DirectForwardPrefillRequest.tokenIds(
                        inputIds,
                        weights,
                        config,
                        arch,
                        kvCache,
                        resolvedWeights));
    }

    public AccelTensor prefillLogitsTensor(AccelTensor embeddings, long[] inputIds, AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache) {
        return prefillLogitsTensor(embeddings, inputIds, perLayerInputs, weights, config, arch, kvCache,
                modelContext.resolveWeights(weights, config, arch));
    }

    private AccelTensor prefillLogitsTensor(AccelTensor embeddings, long[] inputIds, AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache, ResolvedModelWeights resolvedWeights) {
        return prefillLogitsTensor(embeddings, inputIds, perLayerInputs, weights, config, arch, kvCache,
                resolvedWeights, false);
    }

    private AccelTensor prefillLogitsTensor(AccelTensor embeddings, long[] inputIds, AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache, ResolvedModelWeights resolvedWeights,
            boolean embeddingsAlreadyInWorkspace) {
        DirectForwardRuntimeContext runtime = runtimeContext();
        return DirectForwardSequenceRunner.prefillEmbeddings(
                sequenceContext(runtime, config, arch),
                DirectForwardPrefillRequest.embeddings(
                        embeddings,
                        inputIds,
                        perLayerInputs,
                        weights,
                        config,
                        arch,
                        kvCache,
                        resolvedWeights,
                        embeddingsAlreadyInWorkspace));
    }

    public float[] decode(long tokenId, int startPos, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor logits = decodeLogitsTensor(tokenId, startPos, weights, config, arch, kvCache);
        return DirectForwardLogits.materializeAndClose(logits);
    }

    public AccelTensor decodeLogitsTensor(long tokenId, int startPos, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        return decodeLogitsTensor(tokenId, startPos, weights, config, arch, kvCache, false);
    }

    public AccelTensor decodeLogitsTensor(long tokenId, int startPos, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache,
            boolean reuseLogitsOutput) {
        ResolvedModelWeights resolvedWeights = modelContext.resolveWeights(weights, config, arch);
        DirectForwardRuntimeContext runtime = runtimeContext();
        return DirectForwardSequenceRunner.decodeToken(
                sequenceContext(runtime, config, arch),
                new DirectForwardDecodeRequest(
                        tokenId,
                        startPos,
                        weights,
                        config,
                        arch,
                        kvCache,
                        resolvedWeights,
                        reuseLogitsOutput));
    }

    public AccelTensor swigluFfn(AccelTensor x, ModelArchitecture arch, ModelConfig config, AccelTensor gateW, AccelTensor gateB, AccelTensor upW, AccelTensor upB,
            AccelTensor downW, AccelTensor downB) {
        return swigluFfn(x, arch, config, gateW, gateB, upW, upB, downW, downB, null);
    }

    public AccelTensor swigluFfn(AccelTensor x, ModelArchitecture arch, ModelConfig config, AccelTensor gateW, AccelTensor gateB, AccelTensor upW, AccelTensor upB,
            AccelTensor downW, AccelTensor downB, ForwardWorkspace ws) {
        return swigluFfn(x, arch, config, gateW, gateB, upW, upB, downW, downB, ws, null);
    }

    public AccelTensor swigluFfn(AccelTensor x, ModelArchitecture arch, ModelConfig config, AccelTensor gateW, AccelTensor gateB, AccelTensor upW, AccelTensor upB,
            AccelTensor downW, AccelTensor downB, ForwardWorkspace ws,
            AccelTensor downOutputBuffer) {
        return ffnService.swigluFfn(x, arch, config, gateW, gateB, upW, upB, downW, downB, ws, downOutputBuffer);
    }

    // ── Embedding ─────────────────────────────────────────────────────

    public AccelTensor embeddingLookup(AccelTensor embedTable, long[] tokenIds) {
        return DirectForwardTensorOps.embeddingLookup(embedTable, tokenIds);
    }

    private DirectForwardRuntimeContext runtimeContext() {
        return runtimeState.context(log);
    }

    private DirectForwardOperators operators(DirectForwardRuntimeContext runtime) {
        return new DirectForwardOperators(runtime, modelContext::traits);
    }

    private DirectForwardSequenceContext sequenceContext(
            DirectForwardRuntimeContext runtime,
            ModelConfig config,
            ModelArchitecture arch) {
        return new DirectForwardSequenceContext(
                runtime,
                attentionKernel,
                moeForwardPass,
                modelContext.traits(config, arch),
                operators(runtime));
    }

}
