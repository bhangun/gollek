/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * MoeForwardPass.java
 * ────────────────────
 * Sparse Mixture-of-Experts FFN layer for Mixtral 8×7B, Mixtral 8×22B,
 * DeepSeek-MoE, and Grok-1.
 *
 * Now uses AccelTensor + Apple Accelerate. No LibTorch dependency.
 */
package tech.kayys.gollek.safetensor.engine.generation.moe;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;

import java.util.*;

/**
 * Sparse mixture-of-experts FFN computation using AccelTensor.
 */
@ApplicationScoped
public class MoeForwardPass {

    private static final Logger log = Logger.getLogger(MoeForwardPass.class);

    @Inject
    DirectForwardPass forwardPass;

    /**
     * Compute the MoE FFN for one transformer layer using AccelTensor.
     */
    public AccelTensor computeAccel(AccelTensor hidden, Map<String, AccelTensor> weights,
            ModelConfig config, int layerIdx) {

        int numExperts = config.numLocalExperts();
        int topK = config.numExpertsPerTok();

        log.tracef("MoE layer %d: %d experts, top-%d routing", layerIdx, numExperts, topK);

        // 1. Router
        String gateKey = "model.layers.%d.block_sparse_moe.gate.weight".formatted(layerIdx);
        AccelTensor gateWeight = weights.get(gateKey);
        if (gateWeight == null) {
            gateKey = "model.layers.%d.mlp.gate.weight".formatted(layerIdx);
            gateWeight = weights.get(gateKey);
        }
        if (gateWeight == null) {
            log.warnf("MoE router weight not found for layer %d — using expert 0 only", layerIdx);
            return runExpert(hidden, weights, layerIdx, 0);
        }

        // router_logits = hidden_flat @ gateWeight^T
        AccelTensor hiddenFlat = flattenBatch(hidden);
        AccelTensor routerLogits = AccelOps.linear(hiddenFlat, gateWeight);

        // 2. Softmax
        AccelTensor routerProbs = AccelOps.softmax(routerLogits, -1);
        routerLogits.close();

        // 3. Top-K selection
        float[] probData = routerProbs.toFloatArray();
        long[] shape = routerProbs.shape();
        int numTokens = (int) shape[0];
        int numExpertsInWeights = (int) shape[1];
        routerProbs.close();

        int[] expertIndices = new int[numTokens * topK];
        float[] expertWeights = new float[numTokens * topK];
        selectTopK(probData, numTokens, numExpertsInWeights, topK, expertIndices, expertWeights);

        // 4. Compute per-expert and accumulate
        float[] hiddenData = hiddenFlat.toFloatArray();
        int hiddenSize = config.hiddenSize();
        float[] outputData = new float[numTokens * hiddenSize];

        for (int t = 0; t < numTokens; t++) {
            for (int ki = 0; ki < topK; ki++) {
                int expIdx = expertIndices[t * topK + ki];
                float expWeight = expertWeights[t * topK + ki];
                if (expIdx >= numExperts) continue;

                float[] tokenHidden = Arrays.copyOfRange(
                        hiddenData, t * hiddenSize, (t + 1) * hiddenSize);

                float[] expertOut = runExpertOnToken(tokenHidden, weights, layerIdx, expIdx);

                for (int d = 0; d < hiddenSize; d++) {
                    outputData[t * hiddenSize + d] += expWeight * expertOut[d];
                }
            }
        }

        if (hiddenFlat != hidden) hiddenFlat.close();

        // 5. Reshape back
        long[] origShape = hidden.shape();
        return AccelTensor.fromFloatArray(outputData, origShape[0], origShape[1], hiddenSize);
    }

    private static void selectTopK(float[] probs, int numTokens, int numExperts, int k,
            int[] outIndices, float[] outWeights) {
        Integer[] idx = new Integer[numExperts];
        for (int t = 0; t < numTokens; t++) {
            int base = t * numExperts;
            for (int e = 0; e < numExperts; e++) idx[e] = e;
            final int fb = base;
            Arrays.sort(idx, (a, b) -> Float.compare(probs[fb + b], probs[fb + a]));

            float weightSum = 0f;
            for (int ki = 0; ki < k && ki < numExperts; ki++) {
                outIndices[t * k + ki] = idx[ki];
                outWeights[t * k + ki] = probs[base + idx[ki]];
                weightSum += probs[base + idx[ki]];
            }
            if (weightSum > 0f) {
                for (int ki = 0; ki < k; ki++) {
                    outWeights[t * k + ki] /= weightSum;
                }
            }
        }
    }

    private AccelTensor runExpert(AccelTensor hidden, Map<String, AccelTensor> weights,
            int layerIdx, int expertIdx) {
        String prefix = "model.layers.%d.block_sparse_moe.experts.%d".formatted(layerIdx, expertIdx);
        AccelTensor gateW = weights.get(prefix + ".w1.weight");
        AccelTensor upW = weights.get(prefix + ".w3.weight");
        AccelTensor downW = weights.get(prefix + ".w2.weight");
        if (gateW == null) {
            gateW = weights.get("model.layers.%d.mlp.experts.%d.gate_proj.weight".formatted(layerIdx, expertIdx));
            upW = weights.get("model.layers.%d.mlp.experts.%d.up_proj.weight".formatted(layerIdx, expertIdx));
            downW = weights.get("model.layers.%d.mlp.experts.%d.down_proj.weight".formatted(layerIdx, expertIdx));
        }
        if (gateW == null || upW == null || downW == null) {
            log.warnf("Expert %d weights not found for layer %d", expertIdx, layerIdx);
            return hidden;
        }
        return forwardPass.swigluFfn(hidden, gateW, null, upW, null, downW, null);
    }

    private float[] runExpertOnToken(float[] tokenHidden, Map<String, AccelTensor> weights,
            int layerIdx, int expertIdx) {
        try {
            int hiddenSize = tokenHidden.length;
            AccelTensor tokenTensor = AccelTensor.fromFloatArray(tokenHidden, 1, 1, hiddenSize);
            AccelTensor out = runExpert(tokenTensor, weights, layerIdx, expertIdx);
            float[] result = out.toFloatArray();
            tokenTensor.close();
            if (out != tokenTensor) out.close();
            return result;
        } catch (Exception e) {
            log.warnf(e, "Expert %d computation failed at layer %d", expertIdx, layerIdx);
            return tokenHidden;
        }
    }

    private static AccelTensor flattenBatch(AccelTensor t) {
        long[] s = t.shape();
        if (s.length == 2) return t;
        long tokens = s[0] * s[1];
        return t.reshape(tokens, s[s.length - 1]);
    }
}
