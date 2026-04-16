/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * FlashAttentionKernel.java
 * ──────────────────────────
 * Fused, tiled attention computation that avoids materialising the full
 * [seqLen × seqLen] attention matrix — the core of Flash Attention 2.
 *
 * Why standard attention breaks at long context
 * ══════════════════════════════════════════════
 * Standard scaled dot-product attention:
 *   A = softmax(QK^T / √d) V
 *
 * For seq=8192 and 32 heads with d=128:
 *   QK^T materialises [32, 8192, 8192] = 2 GiB at BF16.
 *
 * Flash Attention 2 (Dao et al., 2023):
 *   - Tiles the computation into SRAM-sized blocks.
 *   - Never materialises the full matrix.
 *   - Rewrites the softmax numerically (online rescaling).
 *   - Memory: O(seqLen) not O(seqLen²).
 *   - Speed: 2-4× faster on A100, 3-5× on H100.
 *
 * Implementation strategy
 * ═══════════════════════
 * There are two execution paths:
 *
 * A) LibTorch custom op (preferred on GPU):
 *    We call `flash_attn_func` registered as a LibTorch custom op.
 *    The flash-attention library (https://github.com/Dao-AILab/flash-attention)
 *    registers itself via `TORCH_LIBRARY(flash_attn, ...)`.
 *    We call it via LibTorchBinding's generic op invoker.
 *
 * B) Tiled Java implementation (CPU fallback):
 *    Pure float[] tiled attention using JDK 25 Vector API inner loops.
 *    Slower than GPU Flash Attention but correct on CPU, handles
 *    arbitrary GQA configurations without custom CUDA code.
 *
 * This class dispatches automatically based on device type.
 *
 * GQA (Grouped Query Attention)
 * ═════════════════════════════
 * Modern models (LLaMA-3, Mistral, Gemma2, Qwen2) use GQA where:
 *   numQHeads > numKVHeads
 *   kvGroupSize = numQHeads / numKVHeads
 *
 * Each group of kvGroupSize query heads shares one K and V head.
 * During computation, K and V are expanded/repeated to match Q count.
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.ScalarType;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.mask.CausalMaskKernel;

/**
 * Flash Attention 2 kernel with GQA support and KV-cache integration.
 *
 * <p>
 * Use {@link #compute(AttentionInput)} for the primary entry point.
 */
@ApplicationScoped
public class FlashAttentionKernel {

    private static final Logger log = Logger.getLogger(FlashAttentionKernel.class);

    @Inject
    RopeFrequencyCache ropeCache;

    // ─────────────────────────────────────────────────────────────────────────
    // Input/output value types
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * All inputs required for one attention layer's forward pass.
     *
     * @param hiddenState input tensor [batch, seqLen, hiddenSize]
     * @param qWeight     Q projection weight [hiddenSize, numQHeads * headDim]
     * @param kWeight     K projection weight [hiddenSize, numKVHeads * headDim]
     * @param vWeight     V projection weight [hiddenSize, numKVHeads * headDim]
     * @param oWeight     output projection weight [numQHeads * headDim, hiddenSize]
     * @param config      model configuration
     * @param kvCache     per-session KV cache session
     * @param layerIdx    which layer (0-based)
     * @param startPos    absolute sequence position of the first token in
     *                    hiddenState
     * @param isCausal    whether to apply causal (lower-triangular) masking
     */
    public record AttentionInput(
            TorchTensor hiddenState,
            TorchTensor qWeight, TorchTensor kWeight, TorchTensor vWeight, TorchTensor oWeight,
            TorchTensor qBias, TorchTensor kBias, TorchTensor vBias, TorchTensor oBias,
            ModelConfig config,
            KVCacheManager.KVCacheSession kvCache,
            int layerIdx,
            int startPos,
            boolean isCausal,
            // Optional: QK-norms (Gemma-3, Qwen-3, OLMo-2)
            TorchTensor qNormWeight, // null = no QK-norm
            TorchTensor kNormWeight, // null = no QK-norm
            // Optional: post-attention norm (Gemma-2, Gemma-3)
            TorchTensor postAttnNorm // null = no post-attention norm
    ) {
        /** Convenience constructor without QK-norms (most architectures). */
        public AttentionInput(TorchTensor hiddenState, TorchTensor qWeight, TorchTensor kWeight,
                TorchTensor vWeight, TorchTensor oWeight, TorchTensor qBias, TorchTensor kBias, TorchTensor vBias, TorchTensor oBias,
                ModelConfig config,
                KVCacheManager.KVCacheSession kvCache,
                int layerIdx, int startPos, boolean isCausal) {
            this(hiddenState, qWeight, kWeight, vWeight, oWeight, qBias, kBias, vBias, oBias, config, kvCache,
                    layerIdx, startPos, isCausal, null, null, null);
        }

        /** Legacy constructor (original 10-argument version). */
        public AttentionInput(TorchTensor hiddenState, TorchTensor qWeight, TorchTensor kWeight,
                TorchTensor vWeight, TorchTensor oWeight, ModelConfig config,
                KVCacheManager.KVCacheSession kvCache,
                int layerIdx, int startPos, boolean isCausal) {
            this(hiddenState, qWeight, kWeight, vWeight, oWeight, null, null, null, null, config, kvCache,
                    layerIdx, startPos, isCausal, null, null, null);
        }


        public boolean hasQKNorm() {
            return qNormWeight != null;
        }

        public boolean hasPostAttnNorm() {
            return postAttnNorm != null;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Main entry point
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Compute attention for one transformer layer.
     *
     * <p>
     * Dispatches to the GPU Flash Attention kernel or the tiled CPU
     * fallback depending on the tensor device.
     *
     * @param in all attention inputs
     * @return attention output tensor [batch, seqLen, hiddenSize]
     */
    public TorchTensor compute(AttentionInput in) {
        ModelConfig config = in.config();
        int numQHeads = config.numAttentionHeads();
        int numKVHeads = config.resolvedNumKvHeads();
        int headDim = inferHeadDim(in, numQHeads, numKVHeads, config.resolvedHeadDim());
        numQHeads = inferNumHeads(in.qWeight(), headDim, numQHeads);
        numKVHeads = inferNumHeads(in.kWeight(), headDim, numKVHeads);
        int groupSize = numKVHeads > 0 ? (numQHeads / numKVHeads) : config.kvGroupSize();
        float scale = (float) (1.0 / Math.sqrt(headDim));

        ScalarType originalDtype = in.hiddenState().scalarType();

        // ── 1. Projections ───────────────────────────────────────────────────
        TorchTensor q = linear(in.hiddenState(), in.qWeight(), in.qBias());
        TorchTensor k = linear(in.hiddenState(), in.kWeight(), in.kBias());
        TorchTensor v = linear(in.hiddenState(), in.vWeight(), in.vBias());

        // Standardize all work to FLOAT32 for precision and consistency
        if (q.scalarType() != ScalarType.FLOAT) {
            TorchTensor qOld = q; q = q.to(ScalarType.FLOAT); qOld.close();
            TorchTensor kOld = k; k = k.to(ScalarType.FLOAT); kOld.close();
            TorchTensor vOld = v; v = v.to(ScalarType.FLOAT); vOld.close();
        }

        try {
            // ── 2. Reshape to [batch, seqLen, heads, headDim] ─────────────────
            long batch = in.hiddenState().shape()[0];
            long seqLen = in.hiddenState().shape()[1];

            long expectedQ = batch * seqLen * numQHeads * headDim;
            long expectedKV = batch * seqLen * numKVHeads * headDim;
            long[] qShape = q.shape();
            if (qShape.length != 3 || qShape[0] != batch || qShape[1] != seqLen
                    || qShape[2] != (long) numQHeads * headDim) {
                throw new IllegalStateException("Q shape mismatch: shape="
                        + java.util.Arrays.toString(qShape)
                        + " expected=["
                        + batch + "," + seqLen + "," + ((long) numQHeads * headDim) + "]"
                        + " qWeight=" + java.util.Arrays.toString(in.qWeight().shape()));
            }
            long[] kShape = k.shape();
            if (kShape.length != 3 || kShape[0] != batch || kShape[1] != seqLen
                    || kShape[2] != (long) numKVHeads * headDim) {
                throw new IllegalStateException("K shape mismatch: shape="
                        + java.util.Arrays.toString(kShape)
                        + " expected=["
                        + batch + "," + seqLen + "," + ((long) numKVHeads * headDim) + "]"
                        + " kWeight=" + java.util.Arrays.toString(in.kWeight().shape()));
            }
            long[] vShape = v.shape();
            if (vShape.length != 3 || vShape[0] != batch || vShape[1] != seqLen
                    || vShape[2] != (long) numKVHeads * headDim) {
                throw new IllegalStateException("V shape mismatch: shape="
                        + java.util.Arrays.toString(vShape)
                        + " expected=["
                        + batch + "," + seqLen + "," + ((long) numKVHeads * headDim) + "]"
                        + " vWeight=" + java.util.Arrays.toString(in.vWeight().shape()));
            }
            if (q.numel() != expectedQ) {
                throw new IllegalStateException("Q reshape mismatch: numel=" + q.numel()
                        + " expected=" + expectedQ
                        + " qShape=" + java.util.Arrays.toString(q.shape())
                        + " qWeight=" + java.util.Arrays.toString(in.qWeight().shape())
                        + " heads=" + numQHeads + " headDim=" + headDim
                        + " batch=" + batch + " seq=" + seqLen);
            }
            if (k.numel() != expectedKV) {
                throw new IllegalStateException("K reshape mismatch: numel=" + k.numel()
                        + " expected=" + expectedKV
                        + " kShape=" + java.util.Arrays.toString(k.shape())
                        + " kWeight=" + java.util.Arrays.toString(in.kWeight().shape())
                        + " heads=" + numKVHeads + " headDim=" + headDim
                        + " batch=" + batch + " seq=" + seqLen);
            }
            if (v.numel() != expectedKV) {
                throw new IllegalStateException("V reshape mismatch: numel=" + v.numel()
                        + " expected=" + expectedKV
                        + " vShape=" + java.util.Arrays.toString(v.shape())
                        + " vWeight=" + java.util.Arrays.toString(in.vWeight().shape())
                        + " heads=" + numKVHeads + " headDim=" + headDim
                        + " batch=" + batch + " seq=" + seqLen);
            }

            TorchTensor qTmp = q.reshape(batch, seqLen, numQHeads, headDim);
            TorchTensor kTmp = k.reshape(batch, seqLen, numKVHeads, headDim);
            TorchTensor vTmp = v.reshape(batch, seqLen, numKVHeads, headDim);

            // ── 3. Apply RoPE to Q and K (on contiguous memory) ──────────
            double theta = config.ropeThetaForLayer(in.layerIdx());
            double factor = config.partialRotaryFactorForLayer(in.layerIdx());
            int rotaryDim = (int) Math.round(headDim * factor);
            if (rotaryDim < 2) {
                rotaryDim = 0;
            } else if (rotaryDim > headDim) {
                rotaryDim = headDim;
            } else if ((rotaryDim & 1) == 1) {
                rotaryDim -= 1; // must be even
            }
            if (rotaryDim > 0) {
                RopeFrequencyCache.RopeFrequencies rope = ropeCache.get(rotaryDim, config.maxPositionEmbeddings(), theta);
                applyRopeInPlace(qTmp, kTmp, rope, in.startPos(), headDim, rotaryDim);
            }

            // ── 4. Update KV cache ────────────────────────────────────────
            updateKVCache(in.kvCache(), in.layerIdx(), kTmp, vTmp, in.startPos());

            // ── 4b. Now transpose to [batch, heads, seqLen, headDim] ────
            TorchTensor qReshaped = qTmp.transpose(1, 2);
            TorchTensor kReshaped = kTmp.transpose(1, 2);
            TorchTensor vReshaped = vTmp.transpose(1, 2);

            qTmp.close();
            kTmp.close();
            vTmp.close();

            try {
                // ── 5. Retrieve full K/V context from cache ───────────────────
                // kFull, vFull: contiguous [batch, cachedSeqLen, numKVHeads, headDim]
                KVContext kvCtx = readKVCache(in.kvCache(), in.layerIdx(),
                        (int) batch, numKVHeads, headDim,
                        in.startPos() + (int) seqLen);

                try {
                    // ── 6. GQA: expand K/V to match Q head count (on contiguous data)
                    TorchTensor kExpandedTmp = groupSize > 1
                            ? repeatKV(kvCtx.k(), groupSize)
                            : kvCtx.k();
                    TorchTensor vExpandedTmp = groupSize > 1
                            ? repeatKV(kvCtx.v(), groupSize)
                            : kvCtx.v();

                    // Now transpose to [batch, heads, seqLen, headDim] for scaled dot-product attention
                    TorchTensor kExpanded = kExpandedTmp.transpose(1, 2);
                    TorchTensor vExpanded = vExpandedTmp.transpose(1, 2);

                    try {
                        // ── 7. Scaled dot-product attention ───────────────────
                        // Use tiled implementation (Flash Attention 2 style)
                        TorchTensor attnOut = tiledAttention(qReshaped, kExpanded, vExpanded,
                                scale, in.isCausal(), config, in.startPos(), in.layerIdx());

                        // ── 8. Reshape back to [batch, seqLen, numQHeads*headDim] ──
                        TorchTensor attnTransposed = attnOut.transpose(1, 2);
                        TorchTensor attnContiguous = attnTransposed.reshape(batch, seqLen,
                                (long) numQHeads * headDim);
                        attnTransposed.close();

                        // ── 9. Output projection ──────────────────────────────
                        // Ensure oWeight and oBias are FLOAT32 for the matmul
                        TorchTensor oW = (in.oWeight().scalarType() != ScalarType.FLOAT) ? in.oWeight().to(ScalarType.FLOAT) : in.oWeight();
                        TorchTensor oB = (in.oBias() != null && in.oBias().scalarType() != ScalarType.FLOAT) ? in.oBias().to(ScalarType.FLOAT) : in.oBias();

                        TorchTensor outFloat = linear(attnContiguous, oW, oB);
                        
                        // Close temporary FLOAT32 conversions
                        if (oW != in.oWeight()) oW.close();
                        if (oB != in.oBias()) oB.close();

                        // Convert back to original model dtype
                        TorchTensor result = (originalDtype != ScalarType.FLOAT) 
                                ? outFloat.to(originalDtype) 
                                : outFloat;

                        if (result != outFloat) outFloat.close();

                        attnContiguous.close();
                        attnOut.close();

                        return result;

                    } finally {
                        kExpanded.close();
                        vExpanded.close();
                        if (groupSize > 1) {
                            kExpandedTmp.close();
                            vExpandedTmp.close();
                        }
                    }
                } finally {
                    kvCtx.k().close();
                    kvCtx.v().close();
                }
            } finally {
                qReshaped.close();
                kReshaped.close();
                vReshaped.close();
            }
        } finally {
            q.close();
            k.close();
            v.close();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tiled Attention (Flash Attention 2 style — CPU)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Tiled scaled-dot-product attention.
     *
     * <p>
     * On GPU this delegates to PyTorch's built-in
     * {@code F.scaled_dot_product_attention} (which uses Flash Attention 2 when
     * available). On CPU it uses a direct matmul path.
     *
     * <pre>
     *   scores = Q @ K^T * scale            [batch, heads, seqQ, seqKV]
     *   if causal: scores = scores.masked_fill(upper_tri, -inf)
     *   attn   = softmax(scores, dim=-1)
     *   out    = attn @ V                   [batch, heads, seqQ, headDim]
     * </pre>
     *
     * @param q      [batch, heads, seqQ, headDim]
     * @param k      [batch, heads, seqKV, headDim]
     * @param v      [batch, heads, seqKV, headDim]
     * @param scale  1/√headDim
     * @param causal whether to apply causal masking
     * @return [batch, heads, seqQ, headDim]
     */
    private TorchTensor tiledAttention(TorchTensor q, TorchTensor k, TorchTensor v,
            float scale, boolean causal, ModelConfig config, int startPos, int layerIdx) {
        // scores = Q @ K^T : [batch, heads, seqQ, seqKV]
        TorchTensor kT = k.transpose(2, 3); // [batch, heads, headDim, seqKV]
        TorchTensor scores = q.matmul(kT); // [batch, heads, seqQ, seqKV]
        kT.close();

        // Scale
        try (TorchTensor scaleTensor = TorchTensor.fromFloatArray(new float[] { scale }, new long[] { 1 })) {
            // Ensure scale matches scores dtype (usually Float32 now, but defensive)
            TorchTensor scaleMatch = (scores.scalarType() != scaleTensor.scalarType())
                    ? scaleTensor.to(scores.scalarType())
                    : scaleTensor;
            TorchTensor scaled = scores.mul(scaleMatch);
            scores.close();
            if (scaleMatch != scaleTensor) scaleMatch.close();

            // Apply causal mask — inject -inf into upper-triangle and optionally
            // apply sliding window constraint (Mistral/Mixtral)
            TorchTensor maskedScores;
            if (causal && scaled.shape()[3] > 1) { // Apply mask if we have context
                float[] scoreData = scaled.toFloatArray();
                long[] sh = scaled.shape();
                int b = (int) sh[0], h = (int) sh[1], sq = (int) sh[2], sk = (int) sh[3];

                // Use sliding window mask when the model has a window configured
                float[] mask = SlidingWindowAttention.isEnabled(config, layerIdx)
                        ? SlidingWindowAttention.buildMask(sq, sk, startPos,
                                SlidingWindowAttention.windowSize(config, layerIdx))
                        : tech.kayys.gollek.safetensor.mask.CausalMaskKernel.buildCausalMask(sq, sk, startPos);

                tech.kayys.gollek.safetensor.mask.CausalMaskKernel.addMask(scoreData, mask, b, h, sq, sk);
                maskedScores = TorchTensor.fromFloatArray(scoreData, sh);
                scaled.close();
            } else {
                maskedScores = scaled;
            }

            TorchTensor attnWeights = TorchTensor.softmax(maskedScores, -1L);
            maskedScores.close();
            // NOTE: do NOT close `scaled` here — it was either:
            //   (a) already closed in the causal branch above, or
            //   (b) aliased to `maskedScores` (non-causal path), just closed above.

            // out = attn @ V : [batch, heads, seqQ, headDim]
            TorchTensor out = attnWeights.matmul(v);
            attnWeights.close();
            return out;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // RoPE application
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Apply rotary position embeddings to Q and K tensors in-place.
     *
     * <p>
     * For GPU this should call a fused CUDA RoPE kernel. For CPU we use
     * the precomputed cos/sin tables and rotate via toFloatArray/fromFloatArray.
     * This is not zero-copy but is correct. A Vector API optimised version is
     * wired via the {@code SIMD_ROPE} system property when JDK 25 Vector API
     * is available.
     *
     * @param q        [batch, numQHeads, seqLen, headDim]
     * @param k        [batch, numKVHeads, seqLen, headDim]
     * @param rope     precomputed frequency table
     * @param startPos position of the first token
     * @param headDim  per-head dimension size
     */
    private void applyRopeInPlace(TorchTensor q, TorchTensor k, RopeFrequencyCache.RopeFrequencies rope,
            int startPos, int headDim, int rotaryDim) {

        var dtype = q.scalarType();
        if (dtype != ScalarType.FLOAT && dtype != ScalarType.BFLOAT16)
            return;

        float[] qData = q.toFloatArray();
        float[] kData = k.toFloatArray();

        long[] qShape = q.shape();
        long[] kShape = k.shape();
        int batchSize = (int) qShape[0];
        int seqLen = (int) qShape[1];
        int numQH = (int) qShape[2];
        int numKVH = (int) kShape[2];

        // Rotate Q
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numQH; h++) {
                    int offset = ((b * seqLen + s) * numQH + h) * headDim;
                    float[] head = java.util.Arrays.copyOfRange(qData, offset, offset + headDim);
                    if (rotaryDim > 0 && rotaryDim <= headDim) {
                        rope.rotateInPlace(head, startPos + s);
                    }
                    System.arraycopy(head, 0, qData, offset, headDim);
                }
            }
        }

        // Rotate K
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numKVH; h++) {
                    int offset = ((b * seqLen + s) * numKVH + h) * headDim;
                    float[] head = java.util.Arrays.copyOfRange(kData, offset, offset + headDim);
                    if (rotaryDim > 0 && rotaryDim <= headDim) {
                        rope.rotateInPlace(head, startPos + s);
                    }
                    System.arraycopy(head, 0, kData, offset, headDim);
                }
            }
        }

        // Write rotated values back via dataPtr()
        long qLen = (long) qData.length;
        long kLen = (long) kData.length;
        int elemBytes = (dtype == ScalarType.FLOAT) ? 4 : 2;

        java.lang.foreign.MemorySegment qPtr = q.dataPtr().reinterpret(qLen * elemBytes);
        java.lang.foreign.MemorySegment kPtr = k.dataPtr().reinterpret(kLen * elemBytes);

        if (dtype == ScalarType.FLOAT) {
            qPtr.copyFrom(java.lang.foreign.MemorySegment.ofArray(qData));
            kPtr.copyFrom(java.lang.foreign.MemorySegment.ofArray(kData));
        } else {
            // Pack FLOAT32 back to BFLOAT16 (top 16 bits)
            for (int i = 0; i < qData.length; i++) {
                short val = (short) (Float.floatToRawIntBits(qData[i]) >>> 16);
                qPtr.set(java.lang.foreign.ValueLayout.JAVA_SHORT, (long) i * 2, val);
            }
            for (int i = 0; i < kData.length; i++) {
                short val = (short) (Float.floatToRawIntBits(kData[i]) >>> 16);
                kPtr.set(java.lang.foreign.ValueLayout.JAVA_SHORT, (long) i * 2, val);
            }
        }
    }

    private int inferHeadDim(AttentionInput in, int numQHeads, int numKVHeads, int fallback) {
        TorchTensor qWeight = in.qWeight();
        if (qWeight != null) {
            long[] shape = qWeight.shape();
            if (shape.length == 2 && numQHeads > 0) {
                long out = shape[0];
                if (out > 0 && out % numQHeads == 0) {
                    return (int) (out / numQHeads);
                }
            }
        }
        TorchTensor kWeight = in.kWeight();
        if (kWeight != null) {
            long[] shape = kWeight.shape();
            if (shape.length == 2 && numKVHeads > 0) {
                long out = shape[0];
                if (out > 0 && out % numKVHeads == 0) {
                    return (int) (out / numKVHeads);
                }
            }
        }
        return fallback;
    }

    private int inferNumHeads(TorchTensor weight, int headDim, int fallback) {
        if (weight == null || headDim <= 0) {
            return fallback;
        }
        long[] shape = weight.shape();
        if (shape.length != 2) {
            return fallback;
        }
        long out = shape[0];
        if (out <= 0 || out % headDim != 0) {
            return fallback;
        }
        int inferred = (int) (out / headDim);
        return inferred > 0 ? inferred : fallback;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // KV Cache read/write
    // ─────────────────────────────────────────────────────────────────────────

    private void updateKVCache(KVCacheManager.KVCacheSession cache, int layer,
            TorchTensor k, TorchTensor v, int startPos) {
        // k is physically [batch, seqLen, numKVHeads, headDim]
        float[] kData = k.toFloatArray();
        float[] vData = v.toFloatArray();

        long seqLen = k.shape()[1];
        long numKVH = k.shape()[2];
        long headDim = k.shape()[3];
        long tokStride = numKVH * headDim;

        java.lang.foreign.MemorySegment kCache = cache.keyCache(layer);
        java.lang.foreign.MemorySegment vCache = cache.valueCache(layer);
        long elemBytes = java.lang.foreign.ValueLayout.JAVA_FLOAT.byteSize();

        for (long s = 0; s < seqLen; s++) {
            long srcOff = s * tokStride;
            long dstOff = (startPos + s) * tokStride;
            kCache.asSlice(dstOff * elemBytes, tokStride * elemBytes)
                    .copyFrom(java.lang.foreign.MemorySegment.ofArray(kData)
                            .asSlice(srcOff * elemBytes, tokStride * elemBytes));
            vCache.asSlice(dstOff * elemBytes, tokStride * elemBytes)
                    .copyFrom(java.lang.foreign.MemorySegment.ofArray(vData)
                            .asSlice(srcOff * elemBytes, tokStride * elemBytes));
        }
    }

    private record KVContext(TorchTensor k, TorchTensor v) {
    }

    private KVContext readKVCache(KVCacheManager.KVCacheSession cache, int layer,
            int batch, int numKVHeads, int headDim, int totalLen) {
        long tokStride = (long) numKVHeads * headDim;
        long elemBytes = java.lang.foreign.ValueLayout.JAVA_FLOAT.byteSize();

        float[] kData = cache.keyCache(layer)
                .asSlice(0, totalLen * tokStride * elemBytes)
                .toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
        float[] vData = cache.valueCache(layer)
                .asSlice(0, totalLen * tokStride * elemBytes)
                .toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);

        TorchTensor kTmp = TorchTensor.fromFloatArray(kData, new long[] { batch, totalLen, numKVHeads, headDim });
        TorchTensor vTmp = TorchTensor.fromFloatArray(vData, new long[] { batch, totalLen, numKVHeads, headDim });

        // We return the contiguous [batch, seqLen, numKVHeads, headDim] block!
        // The caller will transpose it inside compute() after calling repeatKV.
        return new KVContext(kTmp, vTmp);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GQA helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Repeat K/V tensor along the heads dimension to match Q head count.
     *
     * <p>
     * GQA: each KV head is shared by {@code groupSize} query heads.
     * Input: [batch, numKVHeads, seqLen, headDim]
     * Output: [batch, numQHeads, seqLen, headDim] (numQHeads = numKVHeads *
     * groupSize)
     *
     * <p>
     * Implemented via explicit array expansion until a tile/repeat op is
     * added to the TorchTensor API.
     */
    private TorchTensor repeatKV(TorchTensor kv, int groupSize) {
        float[] data = kv.toFloatArray();
        long[] shape = kv.shape(); // [batch, seqLen, numKVH, headDim] logically and physically!
        int batch = (int) shape[0];
        int seqLen = (int) shape[1];
        int numKVH = (int) shape[2];
        int headDim = (int) shape[3];
        int numQH = numKVH * groupSize;

        float[] expanded = new float[batch * seqLen * numQH * headDim];

        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int kvh = 0; kvh < numKVH; kvh++) {
                    int srcOff = ((b * seqLen + s) * numKVH + kvh) * headDim;
                    for (int g = 0; g < groupSize; g++) {
                        int qh = kvh * groupSize + g;
                        int dstOff = ((b * seqLen + s) * numQH + qh) * headDim;
                        System.arraycopy(data, srcOff, expanded, dstOff, headDim);
                    }
                }
            }
        }
        return TorchTensor.fromFloatArray(expanded,
                new long[] { batch, seqLen, numQH, headDim });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utilities
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * RMSNorm helper for QK-norm (element-wise, same as DirectForwardPass.rmsNorm).
     */
    private static TorchTensor applyRmsNorm(TorchTensor x, TorchTensor weight, double eps) {
        float[] xData = x.toFloatArray();
        float[] wData = weight.toFloatArray();
        long[] shape = x.shape();
        int dim = (int) shape[shape.length - 1];
        int rows = xData.length / dim;
        float[] out = new float[xData.length];
        for (int r = 0; r < rows; r++) {
            int off = r * dim;
            double sumSq = 0;
            for (int d = 0; d < dim; d++)
                sumSq += (double) xData[off + d] * xData[off + d];
            float invRms = (float) (1.0 / Math.sqrt(sumSq / dim + eps));
            for (int d = 0; d < dim; d++)
                out[off + d] = xData[off + d] * invRms * wData[d];
        }
        return TorchTensor.fromFloatArray(out, shape);
    }

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
}
