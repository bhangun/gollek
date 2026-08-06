/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.metal.binding.MetalFlashAttentionBinding;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FlashAttentionMetalAttentionFactoryTest {

    @Test
    void createsFacadeWithoutEagerNativeBindingLookup() {
        AtomicInteger metalBindingLookups = new AtomicInteger();
        AtomicInteger fa4BindingLookups = new AtomicInteger();
        Supplier<MetalBinding> metalBinding = countedNullMetalBinding(metalBindingLookups);
        Supplier<MetalFlashAttentionBinding> metalFa4 = countedNullFa4Binding(fa4BindingLookups);
        FlashAttentionRoutingPolicy routing = new FlashAttentionRoutingPolicy(
                () -> false, metalBinding, metalFa4, FlashAttentionRoutingOptions.defaults());

        FlashAttentionMetalAttention attention = new FlashAttentionMetalAttentionFactory(
                () -> false,
                metalBinding,
                metalFa4,
                () -> routing).create();

        assertNotNull(attention);
        assertEquals(0, metalBindingLookups.get());
        assertEquals(0, fa4BindingLookups.get());
    }

    @Test
    void rejectsMissingSuppliers() {
        Supplier<MetalBinding> metalBinding = () -> null;
        Supplier<MetalFlashAttentionBinding> metalFa4 = () -> null;
        Supplier<FlashAttentionRoutingPolicy> routing = () -> new FlashAttentionRoutingPolicy(
                () -> false, metalBinding, metalFa4, FlashAttentionRoutingOptions.defaults());

        assertThrows(NullPointerException.class,
                () -> new FlashAttentionMetalAttentionFactory(null, metalBinding, metalFa4, routing));
        assertThrows(NullPointerException.class,
                () -> new FlashAttentionMetalAttentionFactory(() -> false, null, metalFa4, routing));
        assertThrows(NullPointerException.class,
                () -> new FlashAttentionMetalAttentionFactory(() -> false, metalBinding, null, routing));
        assertThrows(NullPointerException.class,
                () -> new FlashAttentionMetalAttentionFactory(() -> false, metalBinding, metalFa4, null));
    }

    private static Supplier<MetalBinding> countedNullMetalBinding(AtomicInteger lookups) {
        return () -> {
            lookups.incrementAndGet();
            return null;
        };
    }

    private static Supplier<MetalFlashAttentionBinding> countedNullFa4Binding(AtomicInteger lookups) {
        return () -> {
            lookups.incrementAndGet();
            return null;
        };
    }
}
