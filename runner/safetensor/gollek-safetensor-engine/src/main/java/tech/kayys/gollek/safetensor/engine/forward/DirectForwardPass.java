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
import tech.kayys.gollek.metal.MetalComputeBackend;
import jakarta.enterprise.inject.Instance;

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
    @Inject
    Instance<MetalComputeBackend> metalBackend;

    public float[] prefill(long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor embedTable = weights.get(arch.embedTokensWeight());
        if (embedTable == null)
            throw new IllegalStateException("Missing embed tokens weight: " + arch.embedTokensWeight());
        AccelTensor embedded = embeddingLookup(embedTable, inputIds);
        float scale = arch.embeddingScaleFactor((int) embedded.size(-1));
        AccelTensor hidden = scale != 1.0f ? AccelOps.mulScalar(embedded, scale) : embedded;
        try {
            if (scale != 1.0f)
                embedded.close();
            return prefill(hidden, inputIds, weights, config, arch, kvCache);
        } finally {
            hidden.closeWithParent();
        }
    }

    public float[] prefill(AccelTensor embeddings, long[] inputIds, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        long seqLen = embeddings.size(1);
        AccelTensor hidden = embeddings;

        // Ensure KV cache has enough space for the entire prefill sequence
        kvCache.ensureCapacity((int) seqLen);

        boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
        for (int i = 0; i < config.numHiddenLayers(); i++) {
            if (verbose) {
                System.err.printf("[DEBUG] Prefill Layer %d/%d start\n", i, config.numHiddenLayers());
                System.err.flush();
            }
            AccelTensor nextHidden = transformerLayer(hidden, inputIds, weights, config, arch, kvCache, i, 0,
                    (int) seqLen);
            if (verbose) {
                System.err.printf("[DEBUG] Prefill Layer %d/%d end\n", i, config.numHiddenLayers());
                System.err.flush();
            }

            if (hidden != embeddings)
                hidden.close();
            hidden = nextHidden;
        }
        kvCache.advance((int) seqLen);

        AccelTensor normed;
        if (metalBackend.isResolvable() && metalBackend.get().deviceName() != null
                && !metalBackend.get().deviceName().contains("CPU")) {
            normed = AccelTensor.zeros(hidden.shape());
            metalBackend.get().rmsNorm(normed.dataSegment(), hidden.dataSegment(),
                    weights.get(arch.finalNormWeight()).dataSegment(), (int) hidden.size(-1),
                    (float) config.rmsNormEps(), arch.addOneToRmsNormWeight());
        } else {
            normed = AccelOps.rmsNorm(hidden, weights.get(arch.finalNormWeight()), config.rmsNormEps(),
                    arch.addOneToRmsNormWeight());
        }
        if (hidden != embeddings)
            hidden.close();

        AccelTensor lastPos = selectLastToken(normed, (int) seqLen);
        normed.close();

        AccelTensor lmHeadW = weights.get(arch.lmHeadWeight());
        if (lmHeadW == null && config.tieWordEmbeddings()) {
            lmHeadW = weights.get(arch.embedTokensWeight()); // weight tying
        }
        if (lmHeadW == null) {
            throw new IllegalStateException(
                    "Missing lm_head weight. Safetensor file might be incomplete or config.tie_word_embeddings is missing.");
        }
        AccelTensor logits = AccelOps.linear(lastPos, lmHeadW);
        lastPos.close();

        float finalSoftCap = arch.defaultFinalSoftCap();
        if (finalSoftCap > 0.0f) {
            AccelTensor scaled = AccelOps.mulScalar(logits, 1.0f / finalSoftCap);
            AccelTensor tanhed = AccelOps.tanh(scaled);
            AccelTensor capped = AccelOps.mulScalar(tanhed, finalSoftCap);
            logits.close();
            scaled.close();
            tanhed.close();
            logits = capped;
        }

        float[] result = logits.toFloatArray();
        logits.close();

        // Diagnostics
        double sum = 0, min = Double.MAX_VALUE, max = Double.MIN_VALUE;
        for (float f : result) {
            sum += f;
            if (f < min)
                min = f;
            if (f > max)
                max = f;
        }
        if (verbose && config.numAttentionHeads() > 0) { // Just a dummy condition to keep it quiet usually
            System.err.printf("[DEBUG] Logits stats: min=%f, max=%f, sum=%f, size=%d\n", min, max, sum, result.length);

            // Print top 5 IDs for debugging
            List<Integer> topIndices = new ArrayList<>();
            for (int i = 0; i < result.length; i++)
                topIndices.add(i);
            topIndices.sort((a, b) -> Float.compare(result[b], result[a]));
            for (int k = 0; k < Math.min(5, topIndices.size()); k++) {
                int id = topIndices.get(k);
                System.err.printf("  Top %d: ID=%d, val=%f\n", k, id, result[id]);
            }
        }

        return result;
    }

    public float[] decode(long tokenId, int startPos, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor embedTable = weights.get(arch.embedTokensWeight());
        if (embedTable == null)
            throw new IllegalStateException("Missing embed tokens weight.");
        AccelTensor embedded = embeddingLookup(embedTable, new long[] { tokenId });
        float scale = arch.embeddingScaleFactor((int) embedded.size(-1));
        AccelTensor hidden = scale != 1.0f ? AccelOps.mulScalar(embedded, scale) : embedded;
        try {
            if (scale != 1.0f)
                embedded.close();
            long[] tokenIds = new long[] { tokenId };

            // Ensure KV cache has space for the next generated token
            kvCache.ensureCapacity(startPos + 1);

            for (int i = 0; i < config.numHiddenLayers(); i++) {
                AccelTensor nextHidden = transformerLayer(hidden, tokenIds, weights, config, arch, kvCache, i, startPos,
                        1);
                hidden.close();
                hidden = nextHidden;
            }
            kvCache.advance(1);

            AccelTensor normed;
            if (metalBackend.isResolvable() && metalBackend.get().deviceName() != null
                    && !metalBackend.get().deviceName().contains("CPU")) {
                normed = AccelTensor.zeros(hidden.shape());
                metalBackend.get().rmsNorm(normed.dataSegment(), hidden.dataSegment(),
                        weights.get(arch.finalNormWeight()).dataSegment(), (int) hidden.size(-1),
                        (float) config.rmsNormEps(), arch.addOneToRmsNormWeight());
            } else {
                normed = AccelOps.rmsNorm(hidden, weights.get(arch.finalNormWeight()), config.rmsNormEps(),
                        arch.addOneToRmsNormWeight());
            }
            hidden.close();

            AccelTensor lmHeadW = weights.get(arch.lmHeadWeight());
            if (lmHeadW == null && config.tieWordEmbeddings()) {
                lmHeadW = weights.get(arch.embedTokensWeight());
            }
            if (lmHeadW == null) {
                throw new IllegalStateException(
                        "Missing lm_head weight. Safetensor file might be incomplete or config.tie_word_embeddings is missing.");
            }
            AccelTensor logits = AccelOps.linear(normed, lmHeadW);
            normed.close();

            // Logit Soft-capping (Gemma-2)
            float finalCap = arch.defaultFinalSoftCap();
            if (finalCap > 0) {
                AccelTensor scaledIn = AccelOps.mulScalar(logits, 1.0f / finalCap);
                AccelTensor capped = AccelOps.tanh(scaledIn);
                AccelTensor scaledOut = AccelOps.mulScalar(capped, finalCap);
                logits.close();
                scaledIn.close();
                capped.close();
                logits = scaledOut;
            }

            float[] result = logits.toFloatArray();
            logits.close();
            return result;
        } finally {
            if (!hidden.isClosed())
                hidden.closeWithParent();
        }
    }

    private AccelTensor transformerLayer(AccelTensor hidden, long[] inputIds, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache, int layerIdx,
            int startPos, int seqLen) {
        AccelTensor residual = hidden;

        boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
        // Attention norm
        AccelTensor normedAttn;
        if (metalBackend.isResolvable() && metalBackend.get().deviceName() != null
                && !metalBackend.get().deviceName().contains("CPU")) {
            normedAttn = AccelTensor.zeros(residual.shape());
            metalBackend.get().rmsNorm(normedAttn.dataSegment(), residual.dataSegment(),
                    weights.get(arch.layerAttentionNormWeight(layerIdx)).dataSegment(), (int) residual.size(-1),
                    (float) config.rmsNormEps(), arch.addOneToRmsNormWeight());
        } else {
            normedAttn = AccelOps.rmsNorm(residual, weights.get(arch.layerAttentionNormWeight(layerIdx)),
                    config.rmsNormEps(), arch.addOneToRmsNormWeight());
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
                arch, config, kvCache, layerIdx, startPos,
                /* isCausal= */ true,
                weights.get(arch.layerQueryNormWeight(layerIdx)),
                weights.get(arch.layerKeyNormWeight(layerIdx)),
                weights.get(arch.layerPostAttnNormWeight(layerIdx)));

        if (verbose) {
            System.err.printf("[DEBUG] Layer %d Attention start\n", layerIdx);
            System.err.flush();
        }
        AccelTensor attnOut = attentionKernel.compute(attnIn);
        if (verbose) {
            System.err.printf("[DEBUG] Layer %d Attention end\n", layerIdx);
            System.err.flush();
        }
        normedAttn.close();

        AccelTensor nextResidual = AccelOps.add(hidden, attnOut);

        AccelTensor normedFfn;
        AccelTensor preFfnNormW = weights.get(arch.layerPreFfnNormWeight(layerIdx));
        if (preFfnNormW == null)
            preFfnNormW = weights.get(arch.layerFfnNormWeight(layerIdx)); // Fallback for older Gemma

        if (metalBackend.isResolvable() && metalBackend.get().deviceName() != null
                && !metalBackend.get().deviceName().contains("CPU")) {
            normedFfn = AccelTensor.zeros(nextResidual.shape());
            metalBackend.get().rmsNorm(normedFfn.dataSegment(), nextResidual.dataSegment(), preFfnNormW.dataSegment(),
                    (int) nextResidual.size(-1), (float) config.rmsNormEps(), arch.addOneToRmsNormWeight());
        } else {
            normedFfn = AccelOps.rmsNorm(nextResidual, preFfnNormW, config.rmsNormEps(), arch.addOneToRmsNormWeight());
        }
        attnOut.close();
        residual = nextResidual;

        AccelTensor ffnOut;
        if (verbose) {
            System.err.printf("[DEBUG] Layer %d FFN start\n", layerIdx);
            System.err.flush();
        }
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
        if (verbose) {
            System.err.printf("[DEBUG] Layer %d FFN end\n", layerIdx);
            System.err.flush();
        }

        normedFfn.close();

        // Optional post-FFN norm (Gemma-2)
        AccelTensor ffnFinal = ffnOut;
        AccelTensor postFfnNormW = weights.get(arch.layerFfnNormWeight(layerIdx));
        if (weights.get(arch.layerPreFfnNormWeight(layerIdx)) != null) {
            // If we had a pre-norm, then layerFfnNormWeight is the post-norm
            ffnFinal = AccelOps.rmsNorm(ffnOut, postFfnNormW, config.rmsNormEps(), arch.addOneToRmsNormWeight());
            ffnOut.close();
        }

        AccelTensor output = AccelOps.add(residual, ffnFinal);
        ffnFinal.close();
        residual.close(); // residual is nextResidual

        return output;
    }

    private void checkStability(AccelTensor t, String label) {
        float[] arr = t.toFloatArray();
        int nan = 0;
        int inf = 0;
        for (float f : arr) {
            if (Float.isNaN(f))
                nan++;
            if (Float.isInfinite(f))
                inf++;
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

    public AccelTensor swigluFfn(AccelTensor x, AccelTensor gateW, AccelTensor gateB, AccelTensor upW, AccelTensor upB,
            AccelTensor downW, AccelTensor downB) {
        AccelTensor gate = AccelOps.linear(x, gateW);
        AccelTensor up = AccelOps.linear(x, upW);

        AccelTensor combined;
        if (metalBackend.isResolvable() && metalBackend.get().deviceName() != null
                && !metalBackend.get().deviceName().contains("CPU")) {
            combined = AccelTensor.zeros(gate.shape());
            metalBackend.get().siluFfn(combined.dataSegment(), gate.dataSegment(), up.dataSegment(),
                    (int) gate.numel());
        } else {
            combined = AccelOps.swiglu(gate, up);
        }

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
