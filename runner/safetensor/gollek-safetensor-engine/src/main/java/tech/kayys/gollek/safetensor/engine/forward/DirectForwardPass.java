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
        
        System.err.println("--- Embeddings (token 0, first 10 dims) ---");
        for(int i=0; i<10; i++) System.err.print(hidden.dataSegment().getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i) + ", ");
        System.err.println();

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
        if (lmHeadW == null) lmHeadW = weights.get(arch.embedTokensWeight()); // weight tying
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
        System.err.println(String.format("[DIAG-KV] Prefill logits: min=%.4f, max=%.4f, avg=%.4f, size=%d", min, max, sum / result.length, result.length));

        return result;
    }

    public float[] decode(long tokenId, int startPos, Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor embedTable = weights.get(arch.embedTokensWeight());
        if (embedTable == null) throw new IllegalStateException("Missing embed tokens weight.");
        AccelTensor hidden = embeddingLookup(embedTable, new long[] { tokenId });
        try {
            long[] tokenIds = new long[] { tokenId };
            for (int i = 0; i < config.numHiddenLayers(); i++) {
                AccelTensor nextHidden = transformerLayer(hidden, tokenIds, weights, config, arch, kvCache, i, startPos, 1);
                hidden.close();
                hidden = nextHidden;
            }
            kvCache.advance(1);

            AccelTensor normed = AccelOps.rmsNorm(hidden, weights.get(arch.finalNormWeight()), config.rmsNormEps());
            hidden.close();

            AccelTensor lmHeadW = weights.get(arch.lmHeadWeight());
            if (lmHeadW == null) lmHeadW = weights.get(arch.embedTokensWeight());
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
        if (layerIdx == 0 && startPos == 0) {
            String keys[] = { arch.layerQueryWeight(0), arch.layerQueryBias(0), arch.layerKeyWeight(0), arch.layerKeyBias(0) };
            for (String k : keys) {
                AccelTensor t = weights.get(k);
                if (t != null) System.err.println("[DIAG-KV] Weight " + k + ": " + t.statistics());
                else System.err.println("[DIAG-KV] Weight " + k + ": NULL");
            }
        }
        AccelTensor residual = hidden;
        if (layerIdx == 0 && startPos == 0) {
            System.err.println("[DIAG-KV] L0_Residual_In: " + residual.statistics());
        }

        // Attention norm
        AccelTensor normedAttn = AccelOps.rmsNorm(residual, weights.get(arch.layerAttentionNormWeight(layerIdx)), config.rmsNormEps());
        
        if (layerIdx == 0 && startPos == 0) {
            System.err.println("[DIAG-KV] L0_Normed_Attn_In: " + normedAttn.statistics());
        }

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
        if (layerIdx == 0 && startPos == 0) {
            System.err.println("[DIAG-KV] L0_Attn_Out: " + attnOut.statistics());
        }
        normedAttn.close();

        AccelTensor afterAttn = AccelOps.add(residual, attnOut);
        attnOut.close();
        
        if (layerIdx == 0 && startPos == 0) {
            System.err.println("[DIAG-KV] L0_Post_Attn_Residual: " + afterAttn.statistics());
        }

        // FFN norm
        AccelTensor residual2 = afterAttn;
        AccelTensor normedFfn = AccelOps.rmsNorm(residual2, weights.get(arch.layerFfnNormWeight(layerIdx)), config.rmsNormEps());

        if (layerIdx == 0 && startPos == 0) {
            System.err.println("[DIAG-KV] L0_Normed_FFN_In: " + normedFfn.statistics());
        }

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
        if (layerIdx == 0 && startPos == 0) {
            System.err.println("[DIAG-KV] L0_FFN_Out: " + ffnOut.statistics());
        }
        normedFfn.close();

        AccelTensor output = AccelOps.add(residual2, ffnOut);
        ffnOut.close();
        residual2.close();

        if (layerIdx == 0 && startPos == 0) {
            System.err.println("[DIAG-KV] L0_Layer_Output: " + output.statistics());
        }

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
            System.err.println(String.format("[DIAG-STABILITY] %s: NaN=%d, Inf=%d, size=%d", label, nan, inf, arr.length));
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
        long start = System.nanoTime();
        AccelTensor gate = AccelOps.linear(x, gateW);
        AccelTensor up = AccelOps.linear(x, upW);
        AccelTensor activated = AccelOps.silu(gate);
        AccelTensor combined = AccelOps.mul(activated, up);
        gate.close();
        up.close();
        activated.close();
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
