/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

final class FlashAttentionRopeStage {
    private final RopeFrequencyCache ropeCache;
    private final FlashAttentionRopePolicy ropePolicy;

    FlashAttentionRopeStage(RopeFrequencyCache ropeCache) {
        this(ropeCache, FlashAttentionRopeOptions.fromSystemProperties());
    }

    FlashAttentionRopeStage(RopeFrequencyCache ropeCache, FlashAttentionRopeOptions options) {
        this.ropeCache = ropeCache;
        this.ropePolicy = FlashAttentionRopePolicy.from(options);
    }

    void apply(AccelTensor q, AccelTensor k, int startPos, ModelConfig config,
            FlashAttentionModelPolicy modelPolicy, int layerIdx, int headDim) {
        // Hugging Face Gemma-4 text uses split-half `rotate_half`, not the
        // legacy adjacent-pair interleaved rotation. Keep the old path only
        // as an explicit escape hatch while we finish the remaining parity
        // work around shared-KV and full/sliding attention interaction.
        boolean interleavedRope = useInterleavedRope(modelPolicy);
        int rotatedDim = resolveRotatedDim(config, layerIdx, headDim);
        int rotaryDim = resolveRotaryStorageDim(headDim, rotatedDim, interleavedRope, modelPolicy);
        RopeFrequencyCache.RopeFrequencies freqs = ropeCache.get(rotaryDim, config.maxPositionEmbeddings(),
                config.ropeThetaForLayer(layerIdx), config.ropeScaling(),
                resolveRopeExponentDenominator(rotaryDim, headDim, config, layerIdx), Math.min(rotaryDim, rotatedDim));
        applyRope(q, k, startPos, freqs, interleavedRope);
    }

    boolean useInterleavedRope(FlashAttentionModelPolicy modelPolicy) {
        return ropePolicy.useInterleavedRope(modelPolicy);
    }

    private void applyRope(AccelTensor q, AccelTensor k, int startPos, RopeFrequencyCache.RopeFrequencies freqs,
            boolean interleaved) {
        int seqLen = (int) q.size(1);
        int numQHeads = (int) q.size(2);
        int headDim = (int) q.size(3);

        for (int s = 0; s < seqLen; s++) {
            int pos = startPos + s;
            for (int h = 0; h < numQHeads; h++) {
                freqs.rotateInPlace(q.dataPtr(), ((long) s * numQHeads + h) * headDim, pos, interleaved);
            }
            if (k != null) {
                int numKVHeads = (int) k.size(2);
                for (int h = 0; h < numKVHeads; h++) {
                    freqs.rotateInPlace(k.dataPtr(), ((long) s * numKVHeads + h) * headDim, pos, interleaved);
                }
            }
        }
    }

    private int resolveRotaryStorageDim(int headDim, int rotatedDim, boolean interleaved,
            FlashAttentionModelPolicy modelPolicy) {
        // Partial RoPE (Gemma-4 proportional full attention) still uses the
        // full head layout. Only the first rotated span receives non-identity
        // frequencies; the remaining split-half pairs must stay in place.
        if (!interleaved && !modelPolicy.nativeBf16Matvec() && rotatedDim < headDim) {
            return rotatedDim;
        }
        return headDim;
    }

    private int resolveRotatedDim(ModelConfig config, int layerIdx, int storageDim) {
        double partialFactor = config.partialRotaryFactorForLayer(layerIdx);
        int rotaryDim = (int) Math.round(storageDim * partialFactor);
        rotaryDim = Math.max(2, rotaryDim);
        if ((rotaryDim & 1) != 0) {
            rotaryDim--;
        }
        return Math.min(storageDim, rotaryDim);
    }

    private int resolveRopeExponentDenominator(int rotaryDim, int headDim, ModelConfig config, int layerIdx) {
        if ("proportional".equals(config.layerType(layerIdx))) {
            return headDim;
        }
        return rotaryDim;
    }
}
