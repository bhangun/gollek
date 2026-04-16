/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * DirectForwardPass.java
 * ───────────────────────
 * The complete transformer forward pass for LLaMA-family models using
 * raw safetensors weights and LibTorch tensor operations.
 *
 * Architecture (LLaMA / Mistral / Qwen2 / Gemma)
 * ══════════════════════════════════════════════════
 *
 *  input_ids [batch, seqLen]
 *      │
 *      ▼  embed_tokens lookup
 *  hidden [batch, seqLen, hiddenSize]
 *      │
 *      └─── for layer i in 0..numLayers:
 *           │
 *           ├── residual = hidden
 *           ├── hidden = rms_norm(hidden, input_layernorm_weight)
 *           ├── hidden = attention(hidden, layer_i_weights, kvCache, pos)
 *           ├── hidden = hidden + residual
 *           │
 *           ├── residual = hidden
 *           ├── hidden = rms_norm(hidden, post_attention_layernorm_weight)
 *           ├── hidden = swiglu_ffn(hidden, gate_proj, up_proj, down_proj)
 *           └── hidden = hidden + residual
 *      │
 *      ▼  rms_norm(hidden, final_norm_weight)
 *      │
 *      ▼  take last position: hidden[:, -1, :]   (for decode)
 *      │
 *      ▼  lm_head(hidden) → logits [batch, vocabSize]
 *      │
 *      ▼  return logits as float[]
 *
 * SwiGLU FFN (used by LLaMA, Mistral, Qwen2):
 *   gate = silu(hidden @ gate_proj^T)
 *   up   = hidden @ up_proj^T
 *   down = (gate ⊙ up) @ down_proj^T
 *
 * Key optimisations in this file:
 *   1. All intermediate tensors are closed immediately after use (no leak).
 *   2. RoPE frequencies are precomputed — no per-step cos/sin computation.
 *   3. Flash Attention is called via FlashAttentionKernel.
 *   4. KV cache is updated in-place per layer.
 */
package tech.kayys.gollek.safetensor.engine.forward;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.inference.libtorch.core.ScalarType;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.attention.FlashAttentionKernel;
import tech.kayys.gollek.safetensor.engine.generation.moe.MoeForwardPass;

import java.util.Map;

/**
 * Full transformer forward pass for LLaMA-family models.
 *
 * <p>
 * This is the P0 implementation that replaces the stubs in
 * {@link tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine}.
 *
 * <p>
 * Call {@link #prefill} or {@link #decode} from the inference engine.
 */
@ApplicationScoped
public class DirectForwardPass {

    private static final Logger log = Logger.getLogger(DirectForwardPass.class);

    @Inject
    FlashAttentionKernel attentionKernel;
    @Inject
    MoeForwardPass moeForwardPass;

    // ─────────────────────────────────────────────────────────────────────────
    // Prefill
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * PREFILL forward pass: process the full prompt in one shot.
     *
     * <p>
     * Populates the KV cache and returns logits for the LAST token position
     * only (used to sample the first generated token).
     *
     * @param inputIds token IDs [promptLen] (1-D, batch=1)
     * @param weights  bridged weight tensors (name → LibTorch TorchTensor)
     * @param config   model config
     * @param arch     resolved architecture for weight name lookup
     * @param kvCache  per-session KV cache (will be filled by this call)
     * @return float[] logits of length vocabSize
     */
    public float[] prefill(long[] inputIds,
            Map<String, TorchTensor> weights,
            ModelConfig config,
            ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache) {

        int seqLen = inputIds.length;
        TorchTensor embedTable = resolveEmbedTable(weights, arch);
        if (embedTable == null) {
            throw new IllegalStateException("Missing embed tokens weight. Expected '" + arch.embedTokensWeight()
                    + "' or a known embed suffix. Available example keys: " + sampleKeys(weights, 8));
        }
        TorchTensor hidden = embeddingLookup(embedTable, inputIds); // [1, seqLen, hiddenSize]

        try {
            return prefill(hidden, inputIds, weights, config, arch, kvCache);
        } finally {
            if (!hidden.isClosed()) hidden.close();
        }
    }

    /**
     * PREFILL forward pass using pre-computed embeddings.
     * 
     * @param embeddings input embeddings [1, seqLen, hiddenSize]
     */
    public float[] prefill(TorchTensor embeddings,
            long[] inputIds,
            Map<String, TorchTensor> weights,
            ModelConfig config,
            ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache) {

        long seqLen = embeddings.shape()[1];
        log.debugf("Prefill (embeddings): seqLen=%d, model=%s", seqLen, config.modelType());

        TorchTensor hidden = embeddings;
        // No lookup needed, embeddings are already provided (e.g. from FusionEngine)

        try {
            // ── 2. Transformer layers ─────────────────────────────────────────
            for (int i = 0; i < config.numHiddenLayers(); i++) {
                TorchTensor nextHidden = transformerLayer(hidden, inputIds, weights, config, arch, kvCache, i,
                        /* startPos= */ 0, (int) seqLen);
                
                // Only if we are NOT the top-level owner (if embeddings came in as a parameter, 
                // we should probably NOT close them, but keep it consistent)
                if (hidden != embeddings) hidden.close();
                hidden = nextHidden;
            }
            kvCache.advance((int) seqLen);

            // ── 3. Final norm ─────────────────────────────────────────────────
            TorchTensor normed = rmsNorm(hidden, weights.get(arch.finalNormWeight()),
                    config.rmsNormEps());
            if (hidden != embeddings) hidden.close();

            // ── 4. Take last token position: [1, hiddenSize] ─────────────────
            TorchTensor lastPos = selectLastToken(normed, (int) seqLen);
            normed.close();

            // ── 5. LM head projection: [1, vocabSize] ────────────────────────
            TorchTensor lmHeadW = resolveLmHead(weights, config, arch);
            TorchTensor logits = linear(lastPos, lmHeadW);
            lastPos.close();

            float[] result = logits.toFloatArray();
            logits.close();
            return result;

        } finally {
             // We don't close 'embeddings' here because we don't own it (it was passed in)
             // But we should clean up if we created local copies.
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Decode
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * DECODE forward pass: single-token step using KV cache.
     *
     * <p>
     * Much cheaper than prefill — computes attention only for the one new
     * token; past context is read from the KV cache.
     *
     * @param tokenId  the most recently generated token ID (scalar)
     * @param startPos absolute position of this token in the sequence
     * @param weights  bridged weight tensors
     * @param config   model config
     * @param arch     architecture for weight names
     * @param kvCache  per-session KV cache (read + appended)
     * @return float[] logits of length vocabSize
     */
    public float[] decode(long tokenId, int startPos,
            Map<String, TorchTensor> weights,
            ModelConfig config,
            ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache) {

        // Embed the single token: [1, 1, hiddenSize]
        TorchTensor embedTable = resolveEmbedTable(weights, arch);
        if (embedTable == null) {
            throw new IllegalStateException("Missing embed tokens weight. Expected '" + arch.embedTokensWeight()
                    + "' or a known embed suffix. Available example keys: " + sampleKeys(weights, 8));
        }
        TorchTensor hidden = embeddingLookup(embedTable, new long[] { tokenId });

        try {
            long[] tokenIds = new long[] { tokenId };
            for (int i = 0; i < config.numHiddenLayers(); i++) {
                TorchTensor nextHidden = transformerLayer(hidden, tokenIds, weights, config, arch, kvCache,
                        i, startPos, /* seqLen= */ 1);
                hidden.close();
                hidden = nextHidden;
            }
            kvCache.advance(1);

            TorchTensor normed = rmsNorm(hidden, weights.get(arch.finalNormWeight()),
                    config.rmsNormEps());
            hidden.close();

            TorchTensor lmHeadW = resolveLmHead(weights, config, arch);
            TorchTensor logits = linear(normed, lmHeadW);
            normed.close();

            float[] result = logits.toFloatArray();
            logits.close();
            return result;

        } finally {
            if (!hidden.isClosed())
                hidden.close();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Transformer layer
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * One complete transformer layer:
     * pre-norm → attention → residual → pre-norm → FFN → residual
     */
    private TorchTensor transformerLayer(TorchTensor hidden,
            long[] inputIds,
            Map<String, TorchTensor> weights,
            ModelConfig config,
            ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache,
            int layerIdx,
            int startPos, int seqLen) {
        // ── Attention sub-layer ───────────────────────────────────────────────
        TorchTensor residual = applyPerLayerInput(hidden, inputIds, weights, config, arch, layerIdx);

        // Pre-attention RMS norm
        TorchTensor normedAttn = rmsNorm(residual,
                weights.get(arch.layerAttentionNormWeight(layerIdx)),
                config.rmsNormEps());

        // Flash Attention (GQA, RoPE, KV cache)
        // Resolve optional QK-norm and post-attention norm weights
        TorchTensor qNormW = resolveOptional(weights, arch, "qNorm", layerIdx);
        TorchTensor kNormW = resolveOptional(weights, arch, "kNorm", layerIdx);
        TorchTensor postAttnW = resolveOptional(weights, arch, "postAttn", layerIdx);

        FlashAttentionKernel.AttentionInput attnIn = new FlashAttentionKernel.AttentionInput(
                normedAttn,
                weights.get(arch.layerQueryWeight(layerIdx)),
                weights.get(arch.layerKeyWeight(layerIdx)),
                weights.get(arch.layerValueWeight(layerIdx)),
                weights.get(arch.layerOutputWeight(layerIdx)),
                resolveBias(weights, arch.layerQueryBias(layerIdx)),
                resolveBias(weights, arch.layerKeyBias(layerIdx)),
                resolveBias(weights, arch.layerValueBias(layerIdx)),
                resolveBias(weights, arch.layerOutputBias(layerIdx)),
                config, kvCache, layerIdx, startPos,
                /* isCausal= */ true,
                qNormW, kNormW, postAttnW);


        TorchTensor attnOut = attentionKernel.compute(attnIn);
        normedAttn.close();

        // Residual add
        TorchTensor afterAttnRaw = residual.add(attnOut);
        attnOut.close();

        // Post-attention norm (Gemma-2, Gemma-3): extra RMSNorm after attention
        // residual
        TorchTensor afterAttn;
        if (postAttnW != null) {
            afterAttn = rmsNorm(afterAttnRaw, postAttnW, config.rmsNormEps());
            afterAttnRaw.close();
        } else {
            afterAttn = afterAttnRaw;
        }

        TorchTensor residual2 = afterAttn;

        // Pre-FFN RMS norm
        // Gemma-2 has both pre_feedforward_layernorm (before gate/up) AND
        // post_feedforward_layernorm (used in layerFfnNormWeight above).
        // The standard layerFfnNormWeight is already applied above to get residual2.
        // For Gemma-2 we need a second norm before the actual FFN projections.
        TorchTensor preFfnNorm = resolveOptional(weights, arch, "preFfn", layerIdx);
        TorchTensor postFfnNorm = weights.get(arch.layerFfnNormWeight(layerIdx));
        TorchTensor normedFfn;
        if (preFfnNorm != null) {
            normedFfn = rmsNorm(residual2, preFfnNorm, config.rmsNormEps());
        } else {
            normedFfn = rmsNorm(residual2, postFfnNorm, config.rmsNormEps());
        }

        // FFN: MoE sparse routing or dense SwiGLU
        TorchTensor ffnOut;
        if (config.isMoeLayer(layerIdx)) {
            ffnOut = moeForwardPass.compute(normedFfn, weights, config, layerIdx);
        } else {
            TorchTensor gW = weights.get(arch.layerFfnGateWeight(layerIdx));
            TorchTensor uW = weights.get(arch.layerFfnUpWeight(layerIdx));
            TorchTensor dW = weights.get(arch.layerFfnDownWeight(layerIdx));

            TorchTensor gB = resolveBias(weights, arch.layerFfnGateBias(layerIdx));
            TorchTensor uB = resolveBias(weights, arch.layerFfnUpBias(layerIdx));
            TorchTensor dB = resolveBias(weights, arch.layerFfnDownBias(layerIdx));

            // Cohere / Command-R: no gate projection — plain SiLU FFN
            ffnOut = (gW == null)
                    ? ffnNonGated(normedFfn, uW, dW, config.hiddenAct())
                    : swigluFfn(normedFfn, gW, gB, uW, uB, dW, dB, config.hiddenAct());

        }
        normedFfn.close();

        // Residual add
        TorchTensor output = residual2.add(ffnOut);
        ffnOut.close();
        residual2.close();

        if (preFfnNorm != null && postFfnNorm != null) {
            TorchTensor post = rmsNorm(output, postFfnNorm, config.rmsNormEps());
            output.close();
            output = post;
        }

        return output;
    }

    private TorchTensor applyPerLayerInput(TorchTensor hidden,
            long[] inputIds,
            Map<String, TorchTensor> weights,
            ModelConfig config,
            ModelArchitecture arch,
            int layerIdx) {
        TorchTensor embedPerLayer = resolvePerLayerEmbedTable(weights, arch);
        TorchTensor gateW = resolveOptional(weights, arch, "perLayerGate", layerIdx);
        TorchTensor projW = resolveOptional(weights, arch, "perLayerProj", layerIdx);
        TorchTensor postNormW = resolveOptional(weights, arch, "perLayerPostNorm", layerIdx);
        int perLayerDim = config.hiddenSizePerLayerInput();

        if (embedPerLayer == null || gateW == null || projW == null || perLayerDim <= 0) {
            return hidden;
        }

        TorchTensor perLayerAll = embeddingLookup(embedPerLayer, inputIds); // [1, seqLen, perLayerDim * layers]
        long start = (long) layerIdx * perLayerDim;
        long end = start + perLayerDim;

        TorchTensor perLayerSlice = selectLastDimRange(perLayerAll, start, end);
        perLayerAll.close();

        TorchTensor gate = linear(hidden, gateW);
        TorchTensor gateAct = tech.kayys.gollek.inference.libtorch.nn.Functional.sigmoid(gate);
        gate.close();

        TorchTensor gated = perLayerSlice.mul(gateAct);
        perLayerSlice.close();
        gateAct.close();

        TorchTensor projected = linear(gated, projW);
        gated.close();

        TorchTensor injected;
        if (postNormW != null) {
            injected = rmsNorm(projected, postNormW, config.rmsNormEps());
            projected.close();
        } else {
            injected = projected;
        }

        TorchTensor out = hidden.add(injected);
        injected.close();
        return out;
    }

    private TorchTensor selectLastDimRange(TorchTensor tensor, long start, long end) {
        int len = (int) (end - start);
        if (len <= 0) {
            throw new IllegalStateException("Invalid slice range: start=" + start + " end=" + end);
        }
        long[] indices = new long[len];
        for (int i = 0; i < len; i++) {
            indices[i] = start + i;
        }
        try (TorchTensor idx = TorchTensor.fromLongArray(indices, new long[] { len })) {
            return tensor.indexSelect(2, idx);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Core operations
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * RMS Normalisation.
     *
     * <pre>
     *   rms   = sqrt(mean(x²) + eps)
     *   x_norm = x / rms * weight
     * </pre>
     *
     * @param x      input [batch, seqLen, hiddenSize]
     * @param weight per-channel scale [hiddenSize]
     * @param eps    stability epsilon
     * @return normalised tensor (same shape as x)
     */
    TorchTensor rmsNorm(TorchTensor x, TorchTensor weight, double eps) {
        // x² mean over last dim → [batch, seqLen, 1]
        TorchTensor xSq = x.mul(x);
        TorchTensor mean = reduceMeanLastDim(xSq);
        xSq.close();

        // rms = 1 / sqrt(mean + eps)
        float epsFlt = (float) eps;
        TorchTensor epsT = TorchTensor.fromFloatArray(new float[] { epsFlt }, new long[] { 1 });
        TorchTensor denom = mean.add(epsT);
        epsT.close();
        mean.close();

        TorchTensor rms = denom.sqrt();
        denom.close();

        // x_norm = x / rms (broadcast over last dim)
        TorchTensor xNorm = x.div(rms);
        rms.close();

        // Scale by learned weight
        TorchTensor output = elementWiseMul(xNorm, weight);
        xNorm.close();
        return output;
    }

    /**
     * Non-gated SiLU FFN for models without a gate projection (Cohere Command-R).
     * Computes: output = down(silu(up(x)))
     */
    TorchTensor ffnNonGated(TorchTensor x, TorchTensor upW, TorchTensor downW, String actName) {
        if (upW == null || downW == null)
            return x;
        TorchTensor up = linear(x, upW);
        TorchTensor act = applyActivation(up, actName);
        up.close();
        TorchTensor out = linear(act, downW);
        act.close();
        return out;
    }

    public TorchTensor swigluFfn(TorchTensor x, TorchTensor gateW, TorchTensor gateB,
                                  TorchTensor upW, TorchTensor upB,
                                  TorchTensor downW, TorchTensor downB,
                                  String actName) {
        TorchTensor gateProj = linear(x, gateW, gateB); // [batch, seq, intermediate]
        TorchTensor gate = applyActivation(gateProj, actName);
        gateProj.close();

        TorchTensor up = linear(x, upW, upB); // [batch, seq, intermediate]
        TorchTensor gated = gate.mul(up); // element-wise product
        gate.close();
        up.close();

        TorchTensor out = linear(gated, downW, downB); // [batch, seq, hiddenSize]
        gated.close();
        return out;
    }


    /**
     * SiLU (Sigmoid Linear Unit): x * sigmoid(x).
     */
    TorchTensor silu(TorchTensor x) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        TorchTensor neg = x.neg();
        TorchTensor expN = neg.exp();
        neg.close();

        TorchTensor one = TorchTensor.fromFloatArray(new float[] { 1.0f }, new long[] { 1 });
        TorchTensor denom = one.add(expN);
        one.close();
        expN.close();

        TorchTensor sig = x.div(denom);
        denom.close();

        // silu = x * sigmoid(x) = x / (1 + exp(-x)) — same as x * sig
        // sig already equals x/denom = x/(1+exp(-x)) ✓
        return sig;
    }

    private TorchTensor applyActivation(TorchTensor x, String actName) {
        String name = (actName == null) ? "" : actName.toLowerCase();
        if (name.contains("gelu")) {
            return tech.kayys.gollek.inference.libtorch.nn.Functional.gelu(x);
        }
        if (name.contains("relu")) {
            return x.relu();
        }
        if (name.contains("tanh")) {
            return tech.kayys.gollek.inference.libtorch.nn.Functional.tanh(x);
        }
        if (name.contains("sigmoid")) {
            return tech.kayys.gollek.inference.libtorch.nn.Functional.sigmoid(x);
        }
        return silu(x);
    }

    /**
     * Embedding lookup: for each token ID, select its row from the embedding
     * matrix.
     *
     * @param embedTable [vocabSize, hiddenSize]
     * @param tokenIds   token IDs
     * @return [1, seqLen, hiddenSize]
     */
    public TorchTensor embeddingLookup(TorchTensor embedTable, long[] tokenIds) {
        if (embedTable == null) {
            throw new IllegalStateException("Embedding lookup failed: embed table is null");
        }
        int seqLen = tokenIds.length;
        
        // FIX: Get device from embedding table and create indices on same device
        tech.kayys.gollek.inference.libtorch.core.Device tableDevice = null;
        try {
            tableDevice = embedTable.getDevice();
        } catch (Exception e) {
            System.err.println("[embeddingLookup] Could not get embedding table device: " + e.getMessage());
            log.warnf("Could not get embedding table device: %s, using default", e.getMessage());
            tableDevice = tech.kayys.gollek.inference.libtorch.core.Device.CPU;
        }
        
        String deviceStr = tableDevice != null ? tableDevice.toString() : "UNKNOWN";
        if (log.isDebugEnabled()) {
            System.out.println("[embeddingLookup] Using device: " + deviceStr + " for " + seqLen + " tokens");
            log.debugf("Embedding lookup: table device=%s, tokenIds=%d", deviceStr, seqLen);
        }
        
        try (TorchTensor idxTensor = TorchTensor.fromLongArray(tokenIds, new long[] { seqLen }, tableDevice)) {
            // indexSelect: select rows from embedTable on dim 0
            // Now indices are on same device as table
            TorchTensor selected = embedTable.indexSelect(0, idxTensor);
            // Reshape to [1, seqLen, hiddenSize]
            long[] shape = selected.shape();
            long hiddenSize = shape[1];
            TorchTensor reshaped = selected.reshape(1L, seqLen, hiddenSize);
            selected.close();
            return reshaped;
        }
    }

    private TorchTensor resolveEmbedTable(Map<String, TorchTensor> weights, ModelArchitecture arch) {
        TorchTensor direct = weights.get(arch.embedTokensWeight());
        if (direct != null) {
            return direct;
        }
        String[] suffixes = {
                "embed_tokens.weight",
                "tok_embeddings.weight",
                "token_embedding.weight"
        };
        TorchTensor best = null;
        for (String key : weights.keySet()) {
            for (String suffix : suffixes) {
                if (key.endsWith(suffix)) {
                    if (best == null) {
                        best = weights.get(key);
                    }
                    if (key.contains("text") || key.contains("language_model")) {
                        return weights.get(key);
                    }
                }
            }
        }
        return best;
    }

    private TorchTensor resolvePerLayerEmbedTable(Map<String, TorchTensor> weights, ModelArchitecture arch) {
        try {
            String name = callStringMethod(arch, "embedTokensPerLayerWeight");
            if (name != null) {
                TorchTensor exact = weights.get(name);
                if (exact != null) return exact;
            }
        } catch (Exception ignored) {
        }

        String suffix = "embed_tokens_per_layer.weight";
        TorchTensor best = null;
        for (String key : weights.keySet()) {
            if (key.endsWith(suffix)) {
                if (best == null) {
                    best = weights.get(key);
                }
                if (key.contains("text") || key.contains("language_model")) {
                    return weights.get(key);
                }
            }
        }
        return best;
    }

    private String sampleKeys(Map<String, TorchTensor> weights, int limit) {
        return weights.keySet().stream().limit(limit).toList().toString();
    }

    /**
     * Select the last token position: hidden[:, -1, :] → [1, hiddenSize].
     */
    private TorchTensor selectLastToken(TorchTensor hidden, int seqLen) {
        // [1, seqLen, hiddenSize] → take seqLen-1 position
        // Use indexSelect on dim=1 with a single-element index tensor
        try (TorchTensor idx = TorchTensor.fromLongArray(new long[] { seqLen - 1L }, new long[] { 1 })) {
            TorchTensor selected = hidden.indexSelect(1, idx); // [1, 1, hiddenSize]
            long hiddenSize = selected.shape()[2];
            TorchTensor squeezed = selected.reshape(1L, hiddenSize); // [1, hiddenSize]
            selected.close();
            return squeezed;
        }
    }

    /**
     * Resolve an optional per-layer weight tensor by reflective arch method name.
     * Returns null (gracefully) if the arch doesn't support the method or the
     * weight is absent from the checkpoint.
     *
     * @param hint one of: "qNorm", "kNorm", "postAttn", "preFfn"
     */
    private TorchTensor resolveOptional(Map<String, TorchTensor> weights,
            ModelArchitecture arch, String hint, int layerIdx) {
        try {
            String name = switch (hint) {
                case "qNorm" -> callStringMethod(arch, "layerQNorm", layerIdx);
                case "kNorm" -> callStringMethod(arch, "layerKNorm", layerIdx);
                case "postAttn" -> callStringMethod(arch, "layerPostAttnNorm", layerIdx);
                case "preFfn" -> callStringMethod(arch, "layerPreFfnNormWeight", layerIdx);
                case "perLayerGate" -> callStringMethod(arch, "layerPerLayerInputGateWeight", layerIdx);
                case "perLayerProj" -> callStringMethod(arch, "layerPerLayerProjectionWeight", layerIdx);
                case "perLayerPostNorm" -> callStringMethod(arch, "layerPostPerLayerInputNormWeight", layerIdx);
                default -> null;
            };
            return name != null ? weights.get(name) : null;
        } catch (Exception e) {
            return null; // method not declared by this arch
        }
    }

    private static String callStringMethod(Object obj, String method, int arg) {
        try {
            return (String) obj.getClass().getMethod(method, int.class).invoke(obj, arg);
        } catch (NoSuchMethodException e) {
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    private static String callStringMethod(Object obj, String method) {
        try {
            return (String) obj.getClass().getMethod(method).invoke(obj);
        } catch (NoSuchMethodException e) {
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    private TorchTensor resolveLmHead(Map<String, TorchTensor> weights,
            ModelConfig config, ModelArchitecture arch) {
        if (config.tieWordEmbeddings() || arch.lmHeadWeight() == null) {
            return weights.get(arch.embedTokensWeight());
        }
        return weights.get(arch.lmHeadWeight());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TorchTensor operation helpers
    // ─────────────────────────────────────────────────────────────────────────

    /** Linear: output = input @ weight^T + bias. */
    private static TorchTensor linear(TorchTensor input, TorchTensor weight, TorchTensor bias) {
        if (input == null || weight == null) {
            throw new IllegalStateException("Linear: input or weight is null");
        }
        TorchTensor in = input;
        TorchTensor w = weight;
        TorchTensor b = bias;
        boolean closeIn = false;
        boolean closeW = false;
        boolean closeB = false;
        if (in.scalarType() != ScalarType.FLOAT) {
            in = in.to(ScalarType.FLOAT);
            closeIn = true;
        }
        if (w.scalarType() != ScalarType.FLOAT) {
            w = w.to(ScalarType.FLOAT);
            closeW = true;
        }
        if (b != null && b.scalarType() != ScalarType.FLOAT) {
            b = b.to(ScalarType.FLOAT);
            closeB = true;
        }
        long[] inShape = input.shape();
        long[] wShape = w.shape();
        if (wShape.length != 2 || inShape.length < 2) {
            throw new IllegalStateException("Linear: invalid shapes input="
                    + java.util.Arrays.toString(inShape)
                    + " weight=" + java.util.Arrays.toString(wShape));
        }
        long inLast = inShape[inShape.length - 1];
        if (inLast != wShape[1]) {
            throw new IllegalStateException("Linear: shape mismatch input(last)=" + inLast
                    + " weight(in)=" + wShape[1]
                    + " inputShape=" + java.util.Arrays.toString(inShape)
                    + " weightShape=" + java.util.Arrays.toString(wShape));
        }
        TorchTensor wT = w.transpose(0, 1);
        TorchTensor mm = in.matmul(wT);
        wT.close();

        if (b != null) {
            TorchTensor out = mm.add(b);
            mm.close();
            if (closeIn) in.close();
            if (closeW) w.close();
            if (closeB) b.close();
            return out;
        }
        if (closeIn) in.close();
        if (closeW) w.close();
        if (closeB) b.close();
        return mm;
    }

    private static TorchTensor linear(TorchTensor input, TorchTensor weight) {
        return linear(input, weight, null);
    }

    private static TorchTensor resolveBias(Map<String, TorchTensor> weights, String key) {
        if (key == null) return null;
        return weights.get(key);
    }


    /**
     * Element-wise multiply, broadcasting weight [hiddenSize] against
     * input [batch, seqLen, hiddenSize].
     */
    private static TorchTensor elementWiseMul(TorchTensor x, TorchTensor w) {
        long hidden = w.shape()[0];
        TorchTensor wReshaped = w.reshape(1L, 1L, hidden);
        TorchTensor result = x.mul(wReshaped);
        wReshaped.close();
        return result;
    }

    /**
     * Reduce mean over the last dimension, keeping dims.
     *
     * <p>
     * This is x.mean(dim=-1, keepdim=True).
     * LibTorch's TorchTensor.mean() reduces all dims. We use a sum/count approach.
     */
    private static TorchTensor reduceMeanLastDim(TorchTensor x) {
        // sum over last dim / count
        long[] shape = x.shape();
        long hiddenSize = shape[shape.length - 1];
        TorchTensor sum = sumLastDim(x);
        TorchTensor countT = TorchTensor.fromFloatArray(new float[] { hiddenSize }, new long[] { 1 });
        try (countT) {
            TorchTensor mean = sum.div(countT);
            sum.close();
            return mean;
        }
    }

    /**
     * Sum over the last dimension via reshape + matmul with all-ones vector.
     *
     * <p>
     * Fallback until a reduce-last-dim op is added to TorchTensor API.
     */
    private static TorchTensor sumLastDim(TorchTensor x) {
        long[] shape = x.shape();
        long hiddenSize = shape[shape.length - 1];
        long outer = x.numel() / hiddenSize;

        TorchTensor flat = x.reshape(outer, hiddenSize);
        if (flat.scalarType() != ScalarType.FLOAT) {
            TorchTensor flatF = flat.to(ScalarType.FLOAT);
            flat.close();
            flat = flatF;
        }
        float[] ones = new float[(int) hiddenSize];
        java.util.Arrays.fill(ones, 1.0f);

         TorchTensor onesT = TorchTensor.fromFloatArray(ones, new long[] { hiddenSize, 1L });
         
         try (onesT) {
             TorchTensor sums = flat.matmul(onesT); // [outer, 1]
             flat.close();

            // Reshape back to original shape but with last dim = 1
            long[] newShape = java.util.Arrays.copyOf(shape, shape.length);
            newShape[shape.length - 1] = 1L;
            TorchTensor result = sums.reshape(newShape);
            sums.close();
            return result;
        }
    }
}
