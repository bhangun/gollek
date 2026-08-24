/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

/**
 * Centralizes model-family normalization and scale decisions for attention.
 */
final class FlashAttentionNormalizationPolicy {
    private final boolean nativeBf16Matvec;
    private final boolean architectureAddsOneToRmsNorm;
    private final double queryPreAttentionScalar;
    private final int headDim;
    private final FlashAttentionNormalizationOptions options;

    private FlashAttentionNormalizationPolicy(boolean nativeBf16Matvec,
            boolean architectureAddsOneToRmsNorm,
            double queryPreAttentionScalar,
            int headDim,
            FlashAttentionNormalizationOptions options) {
        this.nativeBf16Matvec = nativeBf16Matvec;
        this.architectureAddsOneToRmsNorm = architectureAddsOneToRmsNorm;
        this.queryPreAttentionScalar = queryPreAttentionScalar;
        this.headDim = headDim;
        this.options = options;
    }

    static FlashAttentionNormalizationPolicy resolve(ModelArchitecture architecture, ModelConfig config,
            FlashAttentionModelPolicy modelPolicy) {
        return resolve(architecture, config, modelPolicy, FlashAttentionNormalizationOptions.fromSystemProperties());
    }

    static FlashAttentionNormalizationPolicy resolve(ModelArchitecture architecture, ModelConfig config,
            FlashAttentionModelPolicy modelPolicy, FlashAttentionNormalizationOptions options) {
        boolean nativeBf16Matvec = modelPolicy != null && modelPolicy.nativeBf16Matvec();
        boolean addOne = architecture != null && architecture.addOneToRmsNormWeight();
        Double qPas = config == null ? null : config.queryPreAttnScalar();
        double scalar = qPas == null ? 0.0 : qPas;
        int headDim = config == null ? 1 : config.resolvedHeadDim();
        if (options == null) {
            options = FlashAttentionNormalizationOptions.defaults();
        }
        return new FlashAttentionNormalizationPolicy(nativeBf16Matvec, addOne, scalar, headDim, options);
    }

    float attentionScale() {
        if (queryPreAttentionScalar > 0.0) {
            return (float) (1.0 / Math.sqrt(queryPreAttentionScalar));
        } else if (headDim > 0) {
            return (float) (1.0 / Math.sqrt(headDim));
        }
        return 1.0f;
    }

    boolean addOneToRmsNormWeight() {
        return architectureAddsOneToRmsNorm;
    }

    boolean qkNormEnabled() {
        return !nativeBf16Matvec || !options.disableGemma4QkNorm();
    }

    boolean valueNormEnabled() {
        return nativeBf16Matvec && !options.disableGemma4ValueNorm();
    }
}
