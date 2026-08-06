/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.metal.binding.MetalFlashAttentionBinding;

import java.util.Objects;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

final class FlashAttentionMetalAttentionFactory {
    private final BooleanSupplier canUseMetal;
    private final Supplier<MetalBinding> metalBinding;
    private final Supplier<MetalFlashAttentionBinding> metalFa4;
    private final Supplier<FlashAttentionRoutingPolicy> routingPolicy;
    private final FlashAttentionPrecisionOptions precisionOptions;
    private final PagedAttentionVectorOptions pagedAttentionOptions;

    FlashAttentionMetalAttentionFactory(BooleanSupplier canUseMetal, Supplier<MetalBinding> metalBinding,
            Supplier<MetalFlashAttentionBinding> metalFa4,
            Supplier<FlashAttentionRoutingPolicy> routingPolicy) {
        this(canUseMetal, metalBinding, metalFa4, routingPolicy, FlashAttentionPrecisionOptions.defaults(),
                PagedAttentionVectorOptions.defaults());
    }

    FlashAttentionMetalAttentionFactory(BooleanSupplier canUseMetal, Supplier<MetalBinding> metalBinding,
            Supplier<MetalFlashAttentionBinding> metalFa4,
            Supplier<FlashAttentionRoutingPolicy> routingPolicy,
            FlashAttentionPrecisionOptions precisionOptions) {
        this(canUseMetal, metalBinding, metalFa4, routingPolicy, precisionOptions,
                PagedAttentionVectorOptions.defaults());
    }

    FlashAttentionMetalAttentionFactory(BooleanSupplier canUseMetal, Supplier<MetalBinding> metalBinding,
            Supplier<MetalFlashAttentionBinding> metalFa4,
            Supplier<FlashAttentionRoutingPolicy> routingPolicy,
            FlashAttentionPrecisionOptions precisionOptions,
            PagedAttentionVectorOptions pagedAttentionOptions) {
        this.canUseMetal = Objects.requireNonNull(canUseMetal, "canUseMetal");
        this.metalBinding = Objects.requireNonNull(metalBinding, "metalBinding");
        this.metalFa4 = Objects.requireNonNull(metalFa4, "metalFa4");
        this.routingPolicy = Objects.requireNonNull(routingPolicy, "routingPolicy");
        this.precisionOptions = precisionOptions == null ? FlashAttentionPrecisionOptions.defaults() : precisionOptions;
        this.pagedAttentionOptions = pagedAttentionOptions == null ? PagedAttentionVectorOptions.defaults()
                : pagedAttentionOptions;
    }

    FlashAttentionMetalAttention create() {
        FlashAttentionMetalPagedAttention pagedAttention = pagedAttention();
        FlashAttentionMetalFa4GatheredAttention fa4Gathered = fa4GatheredAttention();

        return new FlashAttentionMetalAttention(
                denseSharedAttention(),
                tiledAttention(pagedAttention, fa4Gathered),
                slidingDecodeAttention(pagedAttention));
    }

    private FlashAttentionMetalDenseSharedAttention denseSharedAttention() {
        return new FlashAttentionMetalDenseSharedAttention(canUseMetal, metalBinding, metalFa4, routingPolicy,
                precisionOptions);
    }

    private FlashAttentionMetalTiledAttention tiledAttention(FlashAttentionMetalPagedAttention pagedAttention,
            FlashAttentionMetalFa4GatheredAttention fa4Gathered) {
        return new FlashAttentionMetalTiledAttention(routingPolicy, pagedAttention, fa4Gathered,
                pagedAttentionOptions);
    }

    private FlashAttentionMetalSlidingDecodeAttention slidingDecodeAttention(
            FlashAttentionMetalPagedAttention pagedAttention) {
        return new FlashAttentionMetalSlidingDecodeAttention(metalBinding, routingPolicy, pagedAttention,
                pagedAttentionOptions);
    }

    private FlashAttentionMetalPagedAttention pagedAttention() {
        return new FlashAttentionMetalPagedAttention(metalBinding, routingPolicy);
    }

    private FlashAttentionMetalFa4GatheredAttention fa4GatheredAttention() {
        return new FlashAttentionMetalFa4GatheredAttention(metalFa4, precisionOptions);
    }
}
