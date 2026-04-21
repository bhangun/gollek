/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.attention.FlashAttentionKernel;
import tech.kayys.gollek.safetensor.engine.generation.moe.MoeForwardPass;

import java.util.Map;
import java.util.List;
import java.util.ArrayList;

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

    public float[] prefill(long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor embedTable = weights.get(arch.embedTokensWeight());
        if (embedTable == null) throw new IllegalStateException("Missing embed tokens weight: " + arch.embedTokensWeight());
        AccelTensor hidden = embeddingLookup(embedTable, inputIds);
        try {
            return prefill(hidden, inputIds, weights, config, arch, kvCache);
        } finally {
            hidden.closeWithParent();
        }
    }

    public float[] prefill(AccelTensor embeddings, long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        long seqLen = embeddings.size(1);
        AccelTensor hidden = embeddings;
        
        // Ensure KV cache has enough space for the entire prefill sequence
        kvCache.ensureCapacity((int) seqLen);

        for (int i = 0; i < config.numHiddenLayers(); i++) {
            AccelTensor nextHidden = transformerLayer(hidden, inputIds, weights, config, arch, kvCache, i, 0, (int) seqLen);
            if (hidden != embeddings) hidden.close();
            hidden = nextHidden;
        }
        kvCache.advance((int) seqLen);

        AccelTensor normed = AccelOps.rmsNorm(hidden, weights.get(arch.finalNormWeight()), config.rmsNormEps());
        if (hidden != embeddings) hidden.close();

        AccelTensor lastPos = selectLastToken(normed, (int) seqLen);
        normed.close();

        AccelTensor lmHeadW = weights.get(arch.lmHeadWeight());
        if (lmHeadW == null && config.tieWordEmbeddings()) {
            lmHeadW = weights.get(arch.embedTokensWeight()); // weight tying
        }
        if (lmHeadW == null) {
            throw new IllegalStateException("Missing lm_head weight. Safetensor file might be incomplete or config.tie_word_embeddings is missing.");
        }
        AccelTensor logits = AccelOps.linear(lastPos, lmHeadW);
        lastPos.close();

        float[] result = logits.toFloatArray();
        logits.close();

        // Diagnostics
        double sum = 0, min = Double.MAX_VALUE, max = Double.MIN_VALUE;
        for (float f : result) {
            sum += f;
            if (f < min) min = f;
            if (f > max) max = f;
        }
        if (config.numAttentionHeads() > 0) { // Just a dummy condition to keep it quiet usually
             System.err.printf("[DEBUG] Logits stats: min=%f, max=%f, sum=%f, size=%d\n", min, max, sum, result.length);
        
        // Print top 5 IDs for debugging
        List<Integer> topIndices = new ArrayList<>();
        for (int i = 0; i < result.length; i++) topIndices.add(i);
        topIndices.sort((a, b) -> Float.compare(result[b], result[a]));
        for (int k = 0; k < Math.min(5, topIndices.size()); k++) {
            int id = topIndices.get(k);
            System.err.printf("  Top %d: ID=%d, val=%f\n", k, id, result[id]);
        }
    }

        return result;
    }

    public float[] decode(long tokenId, int startPos, Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor embedTable = weights.get(arch.embedTokensWeight());
        if (embedTable == null) throw new IllegalStateException("Missing embed tokens weight.");
        AccelTensor hidden = embeddingLookup(embedTable, new long[] { tokenId });
        try {
            long[] tokenIds = new long[] { tokenId };
            
            // Ensure KV cache has space for the next generated token
            kvCache.ensureCapacity(startPos + 1);

            for (int i = 0; i < config.numHiddenLayers(); i++) {
                AccelTensor nextHidden = transformerLayer(hidden, tokenIds, weights, config, arch, kvCache, i, startPos, 1);
                hidden.close();
                hidden = nextHidden;
            }
            kvCache.advance(1);

            AccelTensor normed = AccelOps.rmsNorm(hidden, weights.get(arch.finalNormWeight()), config.rmsNormEps());
            hidden.close();

            AccelTensor lmHeadW = weights.get(arch.lmHeadWeight());
            if (lmHeadW == null && config.tieWordEmbeddings()) {
                 lmHeadW = weights.get(arch.embedTokensWeight());
            }
            if (lmHeadW == null) {
                 throw new IllegalStateException("Missing lm_head weight. Safetensor file might be incomplete or config.tie_word_embeddings is missing.");
            }
            AccelTensor logits = AccelOps.linear(normed, lmHeadW);
            normed.close();

            float[] result = logits.toFloatArray();
            logits.close();
            return result;
        } finally {
            if (!hidden.isClosed()) hidden.closeWithParent();
        }
    }

    private AccelTensor transformerLayer(AccelTensor hidden, long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache, int layerIdx, int startPos, int seqLen) {
        AccelTensor residual = hidden;

        // Attention norm
        AccelTensor normedAttn = AccelOps.rmsNorm(residual, weights.get(arch.layerAttentionNormWeight(layerIdx)), config.rmsNormEps());
        
        FlashAttentionKernel.AttentionInput attnIn = new FlashAttentionKernel.AttentionInput(
                normedAttn,
                weights.get(arch.layerQueryWeight(layerIdx)),
                weights.get(arch.layerKeyWeight(layerIdx)),
                weights.get(arch.layerValueWeight(layerIdx)),
                weights.get(arch.layerOutputWeight(layerIdx)),
                resolveBias(weights, arch.layerQueryBias(layerIdx)),
                resolveBias(weights, arch.layerKeyBias(layerIdx)),
                resolveBias(weights, arch.layerValueBias(layerIdx)),
                weights.get(arch.layerOutputBias(layerIdx)),
                config, kvCache, layerIdx, startPos,
                /* isCausal= */ true,
                weights.get(arch.layerQueryNormWeight(layerIdx)),
                weights.get(arch.layerKeyNormWeight(layerIdx)),
                null);

        AccelTensor attnOut = attentionKernel.compute(attnIn);
        normedAttn.close();

        AccelTensor nextResidual = AccelOps.add(residual, attnOut);
        AccelTensor normedFfn = AccelOps.rmsNorm(nextResidual, weights.get(arch.layerFfnNormWeight(layerIdx)), config.rmsNormEps());
        attnOut.close();
        if (residual != hidden) residual.close();
        residual = nextResidual;
        
        AccelTensor ffnOut;
        if (config.isMoeLayer(layerIdx)) {
            ffnOut = moeForwardPass.computeAccel(normedFfn, weights, config, layerIdx);
        } else {
            AccelTensor gW = weights.get(arch.layerFfnGateWeight(layerIdx));
            AccelTensor uW = weights.get(arch.layerFfnUpWeight(layerIdx));
            AccelTensor dW = weights.get(arch.layerFfnDownWeight(layerIdx));
            ffnOut = (gW == null)
                    ? ffnNonGated(normedFfn, uW, dW)
                    : swigluFfn(normedFfn, gW, null, uW, null, dW, null);
        }
        normedFfn.close();

        AccelTensor output = AccelOps.add(residual, ffnOut);
        ffnOut.close();

        return output;
    }

    private void checkStability(AccelTensor t, String label) {
        float[] arr = t.toFloatArray();
        int nan = 0;
        int inf = 0;
        for (float f : arr) {
            if (Float.isNaN(f)) nan++;
            if (Float.isInfinite(f)) inf++;
        }
        if (nan > 0 || inf > 0) {
        }
    }

    AccelTensor ffnNonGated(AccelTensor x, AccelTensor upW, AccelTensor downW) {
        AccelTensor up = AccelOps.linear(x, upW);
        AccelTensor act = AccelOps.silu(up);
        up.close();
        AccelTensor out = AccelOps.linear(act, downW);
        act.close();
        return out;
    }

    public AccelTensor swigluFfn(AccelTensor x, AccelTensor gateW, AccelTensor gateB, AccelTensor upW, AccelTensor upB, AccelTensor downW, AccelTensor downB) {
        AccelTensor gate = AccelOps.linear(x, gateW);
        AccelTensor up = AccelOps.linear(x, upW);
        AccelTensor combined = AccelOps.swiglu(gate, up);
        gate.close();
        up.close();
        AccelTensor out = AccelOps.linear(combined, downW);
        combined.close();
        return out;
    }

    // ── Embedding ─────────────────────────────────────────────────────

    public AccelTensor embeddingLookup(AccelTensor embedTable, long[] tokenIds) {
        int seqLen = tokenIds.length;
        AccelTensor selected = embedTable.indexSelect(tokenIds);
        long hiddenSize = selected.size(1);
        return selected.reshape(1L, seqLen, hiddenSize);
    }

    private AccelTensor selectLastToken(AccelTensor hidden, int seqLen) {
        AccelTensor sliced = hidden.slice(1, seqLen - 1, seqLen);
        long hiddenSize = sliced.size(2);
        return sliced.contiguous().reshape(1L, hiddenSize);
    }

    // ── Linear ────────────────────────────────────────────────────────

    private AccelTensor linear(AccelTensor input, AccelTensor weight, AccelTensor bias) {
        AccelTensor mm = AccelOps.linear(input, weight);
        if (bias != null) {
            AccelTensor out = AccelOps.add(mm, bias);
            mm.close();
            return out;
        }
        return mm;
    }

    private static AccelTensor resolveBias(Map<String, AccelTensor> weights, String key) {
        return key != null ? weights.get(key) : null;
    }
}
