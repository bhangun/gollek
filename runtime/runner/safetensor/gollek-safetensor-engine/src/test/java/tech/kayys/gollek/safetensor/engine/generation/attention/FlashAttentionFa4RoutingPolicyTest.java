/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.metal.binding.MetalFlashAttentionBinding;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class FlashAttentionFa4RoutingPolicyTest {

    @Test
    void rejectsFa4WhenNativeBindingMissing() {
        FlashAttentionFa4RoutingPolicy policy = defaultPolicy(() -> true, () -> null);

        assertFalse(policy.canUseAttention(0.0f));
    }

    @Test
    void rejectsPagedFa4WhenNativeBindingMissing() throws Exception {
        FlashAttentionFa4RoutingPolicy policy = defaultPolicy(() -> true, () -> null);

        assertFalse(policy.canUsePagedAttention(classicConfig(), 0, 0.0f));
    }

    @Test
    void pagedFa4PreservesGlobalMetalShortCircuit() throws Exception {
        AtomicInteger fa4BindingLookups = new AtomicInteger();
        FlashAttentionFa4RoutingPolicy policy = defaultPolicy(
                () -> false,
                () -> {
                    fa4BindingLookups.incrementAndGet();
                    return null;
                });

        assertFalse(policy.canUsePagedAttention(classicConfig(), 0, 0.0f));
        assertEquals(0, fa4BindingLookups.get());
    }

    @Test
    void disabledFa4OptionShortCircuitsBeforeNativeBindingLookup() {
        AtomicInteger fa4BindingLookups = new AtomicInteger();
        FlashAttentionFa4RoutingPolicy policy = new FlashAttentionFa4RoutingPolicy(
                () -> true,
                () -> {
                    fa4BindingLookups.incrementAndGet();
                    return null;
                },
                new FlashAttentionFa4RoutingOptions(true));

        assertFalse(policy.canUseAttention(0.0f));
        assertEquals(0, fa4BindingLookups.get());
    }

    private static FlashAttentionFa4RoutingPolicy defaultPolicy(BooleanSupplier canUseMetal,
            Supplier<MetalFlashAttentionBinding> metalFa4) {
        return new FlashAttentionFa4RoutingPolicy(
                canUseMetal, metalFa4, FlashAttentionFa4RoutingOptions.defaults());
    }

    private static ModelConfig classicConfig() throws Exception {
        return new ObjectMapper().readValue("""
                {
                  "model_type": "phi3",
                  "hidden_size": 256,
                  "intermediate_size": 512,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2
                }
                """, ModelConfig.class);
    }
}
