/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.jboss.logging.Logger;

final class FlashAttentionStageFactory {
    private final Logger log;
    private final RopeFrequencyCache ropeCache;
    private final FlashAttentionStageOptions options;

    FlashAttentionStageFactory(Logger log, RopeFrequencyCache ropeCache) {
        this(log, ropeCache, FlashAttentionStageOptions.fromSystemPropertiesAndEnvironment());
    }

    FlashAttentionStageFactory(Logger log, RopeFrequencyCache ropeCache, FlashAttentionStageOptions options) {
        this.log = log;
        this.ropeCache = ropeCache;
        this.options = options == null ? FlashAttentionStageOptions.defaults() : options;
    }

    FlashAttentionStages initialize() {
        FlashAttentionMetalBindings bindings = FlashAttentionMetalBindings.initialize(log, options.backendOptions());
        FlashAttentionProjector projector = projector(bindings);
        FlashAttentionRoutingPolicy routing = routing(bindings);
        FlashAttentionNormalizer normalizer = normalizer(bindings);
        FlashAttentionDispatchStage dispatch = dispatchStage(bindings, routing);

        return new FlashAttentionStages(
                options,
                new FlashAttentionProjectionStage(projector, normalizer),
                new FlashAttentionRopeStage(ropeCache, options.ropeOptions()),
                dispatch,
                new FlashAttentionKvCacheStage(),
                new FlashAttentionOutputStage(projector, normalizer));
    }

    private FlashAttentionProjector projector(FlashAttentionMetalBindings bindings) {
        return new FlashAttentionProjector(
                bindings.metalBinding(), bindings::canUseMetal, options.linearOptions(), options.matvecOptions());
    }

    private FlashAttentionRoutingPolicy routing(FlashAttentionMetalBindings bindings) {
        return new FlashAttentionRoutingPolicy(bindings::canUseMetal, bindings::metalBinding,
                bindings::metalFa4, options.routingOptions());
    }

    private FlashAttentionNormalizer normalizer(FlashAttentionMetalBindings bindings) {
        return new FlashAttentionNormalizer(bindings::metalBinding, options.normalizerOptions());
    }

    private FlashAttentionDispatchStage dispatchStage(FlashAttentionMetalBindings bindings,
            FlashAttentionRoutingPolicy routing) {
        FlashAttentionMetalAttention metalAttention = new FlashAttentionMetalAttentionFactory(
                bindings::canUseMetal, bindings::metalBinding, bindings::metalFa4, () -> routing,
                options.precisionOptions(), options.pagedAttentionOptions()).create();
        FlashAttentionDispatchRouter dispatchRouter = new FlashAttentionDispatchRouter(routing);
        FlashAttentionDispatchExecutor dispatchExecutor = new FlashAttentionDispatchExecutor(
                metalAttention, options.pagedAttentionOptions());

        return new FlashAttentionDispatchStage(dispatchRouter, dispatchExecutor);
    }
}
