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
 * MoE architecture (Shazeer et al., 2017 / Mixtral paper)
 * ═════════════════════════════════════════════════════════
 * A standard dense FFN layer:
 *   output = down(silu(gate(x)) * up(x))
 *
 * A MoE layer replaces the single FFN with N expert FFNs and a router:
 *   router_logits = x @ gate.weight^T           [batch×seq, N_experts]
 *   top_k_indices, top_k_weights = topK(softmax(router_logits), k=2)
 *   output = Σ_{i in top_k} weight[i] × expert_i(x)
 *
 * For Mixtral 8×7B:
 *   N_experts = 8, top_k = 2
 *   Each expert is a full SwiGLU FFN (gate/up/down projections)
 *   Total weight per layer: 8 × 3 × (4096 × 14336) = 1.4B params
 *   Active weight per token: 2/8 × 1.4B = 350M params
 *
 * Weight naming (Mixtral HuggingFace format)
 * ═══════════════════════════════════════════
 *   model.layers.{i}.block_sparse_moe.gate.weight  [N_experts, hidden_size]
 *   model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight  [intermediate, hidden]
 *   model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight  [hidden, intermediate]
 *   model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight  [intermediate, hidden]
 *   (w1 = gate_proj, w2 = down_proj, w3 = up_proj in SwiGLU terminology)
 *
 * Integration with DirectForwardPass
 * ═══════════════════════════════════
 * DirectForwardPass.runLayer() checks config.isMoeLayer(layerIndex) and
 * delegates to MoeForwardPass.compute() instead of swigluFfn() when true.
 */
package tech.kayys.gollek.safetensor.engine.generation.moe;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;

import java.util.*;

/**
 * Sparse mixture-of-experts FFN computation.
 *
 * <p>
 * Activated automatically by {@link DirectForwardPass} when
 * {@link ModelConfig#isMoeLayer(int)} returns true.
 */
@ApplicationScoped
public class MoeForwardPass {

    private static final Logger log = Logger.getLogger(MoeForwardPass.class);

    @Inject
    DirectForwardPass forwardPass;

    // ─────────────────────────────────────────────────────────────────────────
    // Main entry point
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Compute the MoE FFN for one transformer layer.
     *
     * @param hidden   normed hidden state [batch, seq, hiddenSize]
     * @param weights  full model weight map
     * @param config   model config (numLocalExperts, numExpertsPerToken)
     * @param layerIdx layer index (for weight name construction)
     * @return FFN output [batch, seq, hiddenSize] — same shape as input
     */
    public TorchTensor compute(TorchTensor hidden, Map<String, TorchTensor> weights,
            ModelConfig config, int layerIdx) {

        int numExperts = config.numLocalExperts();
        int topK = config.numExpertsPerTok();

        log.tracef("MoE layer %d: %d experts, top-%d routing", layerIdx, numExperts, topK);

        // ── 1. Router: compute logits for all experts ─────────────────────────
        // gate.weight: [N_experts, hiddenSize]
        String gateKey = "model.layers.%d.block_sparse_moe.gate.weight".formatted(layerIdx);
        TorchTensor gateWeight = weights.get(gateKey);
        if (gateWeight == null) {
            // Fallback: try DeepSeek naming
            gateKey = "model.layers.%d.mlp.gate.weight".formatted(layerIdx);
            gateWeight = weights.get(gateKey);
        }
        if (gateWeight == null) {
            log.warnf("MoE router weight not found for layer %d — using expert 0 only", layerIdx);
            return runExpert(hidden, weights, layerIdx, 0);
        }

        // router_logits = hidden @ gateWeight^T : [batch*seq, N_experts]
        TorchTensor hiddenFlat = flattenBatch(hidden); // [batch*seq, hiddenSize]
        if (hiddenFlat.scalarType() != tech.kayys.gollek.inference.libtorch.core.ScalarType.FLOAT) {
            TorchTensor tmp = hiddenFlat.to(tech.kayys.gollek.inference.libtorch.core.ScalarType.FLOAT);
            hiddenFlat.close();
            hiddenFlat = tmp;
        }
        if (gateWeight.scalarType() != tech.kayys.gollek.inference.libtorch.core.ScalarType.FLOAT) {
            TorchTensor tmp = gateWeight.to(tech.kayys.gollek.inference.libtorch.core.ScalarType.FLOAT);
            gateWeight = tmp;
        }
        try (TorchTensor gateT = gateWeight.transpose(0, 1)) {
            TorchTensor routerLogits = hiddenFlat.matmul(gateT); // [tokens, N_experts]

            // ── 2. Softmax over expert dimension ─────────────────────────────────
            TorchTensor routerProbs = TorchTensor.softmax(routerLogits, -1L);
            routerLogits.close();

            // ── 3. Top-K expert selection ─────────────────────────────────────────
            float[] probData = routerProbs.toFloatArray();
            long[] shape = routerProbs.shape(); // [tokens, N_experts]
            int numTokens = (int) shape[0];
            int numExpertsInWeights = (int) shape[1];
            routerProbs.close();

            // For each token: select top-K expert indices and their weights
            int[] expertIndices = new int[numTokens * topK];
            float[] expertWeights = new float[numTokens * topK];

            selectTopK(probData, numTokens, numExpertsInWeights, topK,
                    expertIndices, expertWeights);

            // ── 4. Compute output for each expert and accumulate ──────────────────
            float[] hiddenData = hiddenFlat.toFloatArray();
            int hiddenSize = config.hiddenSize();
            float[] outputData = new float[numTokens * hiddenSize];

            // Group tokens by expert for batch efficiency (optional optimisation)
            // Simple implementation: iterate over tokens × top-K
            for (int t = 0; t < numTokens; t++) {
                for (int ki = 0; ki < topK; ki++) {
                    int expIdx = expertIndices[t * topK + ki];
                    float expWeight = expertWeights[t * topK + ki];
                    if (expIdx >= numExperts)
                        continue;

                    // Extract single token hidden state
                    float[] tokenHidden = Arrays.copyOfRange(
                            hiddenData, t * hiddenSize, (t + 1) * hiddenSize);

                    // Run this expert's FFN on the token
                    float[] expertOut = runExpertOnToken(tokenHidden, weights, layerIdx, expIdx);

                    // Accumulate weighted output
                    for (int d = 0; d < hiddenSize; d++) {
                        outputData[t * hiddenSize + d] += expWeight * expertOut[d];
                    }
                }
            }

            hiddenFlat.close();

            // ── 5. Reshape back to [batch, seq, hiddenSize] ───────────────────────
            long[] origShape = hidden.shape();
            return TorchTensor.fromFloatArray(outputData,
                    new long[] { origShape[0], origShape[1], hiddenSize });
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Top-K selection
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Select top-K expert indices and weights for each token.
     * Fills expertIndices and expertWeights arrays in-place.
     * Weights are renormalised to sum to 1 among selected experts.
     */
    private static void selectTopK(float[] probs, int numTokens, int numExperts, int k,
            int[] outIndices, float[] outWeights) {
        Integer[] idx = new Integer[numExperts];
        for (int t = 0; t < numTokens; t++) {
            int base = t * numExperts;
            // Sort expert indices by probability descending
            for (int e = 0; e < numExperts; e++)
                idx[e] = e;
            final int fb = base;
            Arrays.sort(idx, (a, b) -> Float.compare(probs[fb + b], probs[fb + a]));

            float weightSum = 0f;
            for (int ki = 0; ki < k && ki < numExperts; ki++) {
                outIndices[t * k + ki] = idx[ki];
                outWeights[t * k + ki] = probs[base + idx[ki]];
                weightSum += probs[base + idx[ki]];
            }
            // Renormalise selected expert weights
            if (weightSum > 0f) {
                for (int ki = 0; ki < k; ki++) {
                    outWeights[t * k + ki] /= weightSum;
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Per-expert FFN
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Run expert j's SwiGLU FFN on the full hidden tensor.
     * Used for the fallback path when only one expert is needed.
     */
    private TorchTensor runExpert(TorchTensor hidden, Map<String, TorchTensor> weights,
            int layerIdx, int expertIdx) {
        String prefix = "model.layers.%d.block_sparse_moe.experts.%d".formatted(layerIdx, expertIdx);
        TorchTensor gateW = weights.get(prefix + ".w1.weight");
        TorchTensor upW = weights.get(prefix + ".w3.weight");
        TorchTensor downW = weights.get(prefix + ".w2.weight");
        if (gateW == null) {
            // DeepSeek naming: gate_proj / up_proj / down_proj
            gateW = weights.get("model.layers.%d.mlp.experts.%d.gate_proj.weight".formatted(layerIdx, expertIdx));
            upW = weights.get("model.layers.%d.mlp.experts.%d.up_proj.weight".formatted(layerIdx, expertIdx));
            downW = weights.get("model.layers.%d.mlp.experts.%d.down_proj.weight".formatted(layerIdx, expertIdx));
        }
        if (gateW == null || upW == null || downW == null) {
            log.warnf("Expert %d weights not found for layer %d", expertIdx, layerIdx);
            return hidden; // passthrough
        }
        return forwardPass.swigluFfn(hidden, gateW, null, upW, null, downW, null, null);
    }

    /**
     * Run expert j's SwiGLU FFN on a single token's hidden state.
     */
    private float[] runExpertOnToken(float[] tokenHidden, Map<String, TorchTensor> weights,
            int layerIdx, int expertIdx) {
        try {
            int hiddenSize = tokenHidden.length;
            TorchTensor tokenTensor = TorchTensor.fromFloatArray(tokenHidden, new long[] { 1, 1, hiddenSize });
            TorchTensor out = runExpert(tokenTensor, weights, layerIdx, expertIdx);
            float[] result = out.toFloatArray();
            tokenTensor.close();
            if (out != tokenTensor)
                out.close();
            return result;
        } catch (Exception e) {
            log.warnf(e, "Expert %d computation failed at layer %d", expertIdx, layerIdx);
            return tokenHidden; // passthrough on error
        }
    }

    /**
     * Flatten [batch, seq, hidden] → [batch*seq, hidden] for router matmul.
     */
    private static TorchTensor flattenBatch(TorchTensor t) {
        long[] s = t.shape();
        if (s.length == 2)
            return t;
        long tokens = s[0] * s[1];
        return t.reshape(tokens, s[s.length - 1]);
    }
}
