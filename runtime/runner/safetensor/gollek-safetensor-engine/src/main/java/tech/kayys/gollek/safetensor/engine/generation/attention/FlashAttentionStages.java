/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.jboss.logging.Logger;

final class FlashAttentionStages {
    private final FlashAttentionStageOptions options;
    private final FlashAttentionProjectionStage projection;
    private final FlashAttentionRopeStage rope;
    private final FlashAttentionDispatchStage dispatch;
    private final FlashAttentionKvCacheStage kvCache;
    private final FlashAttentionOutputStage output;

    FlashAttentionStages(FlashAttentionProjectionStage projection,
            FlashAttentionRopeStage rope,
            FlashAttentionDispatchStage dispatch,
            FlashAttentionKvCacheStage kvCache,
            FlashAttentionOutputStage output) {
        this(FlashAttentionStageOptions.defaults(), projection, rope, dispatch, kvCache, output);
    }

    FlashAttentionStages(FlashAttentionStageOptions options,
            FlashAttentionProjectionStage projection,
            FlashAttentionRopeStage rope,
            FlashAttentionDispatchStage dispatch,
            FlashAttentionKvCacheStage kvCache,
            FlashAttentionOutputStage output) {
        this.options = options == null ? FlashAttentionStageOptions.defaults() : options;
        this.projection = projection;
        this.rope = rope;
        this.dispatch = dispatch;
        this.kvCache = kvCache;
        this.output = output;
    }

    static FlashAttentionStages initialize(Logger log, RopeFrequencyCache ropeCache) {
        return new FlashAttentionStageFactory(log, ropeCache).initialize();
    }

    FlashAttentionStageOptions options() {
        return options;
    }

    FlashAttentionProjectionStage projection() {
        return projection;
    }

    FlashAttentionRopeStage rope() {
        return rope;
    }

    FlashAttentionDispatchStage dispatch() {
        return dispatch;
    }

    FlashAttentionKvCacheStage kvCache() {
        return kvCache;
    }

    FlashAttentionOutputStage output() {
        return output;
    }
}
