/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionMetalRoutingPolicyTest {

    @Test
    void classicAttentionDoesNotProbeBindingsOrFa4Availability() throws Exception {
        AtomicInteger bindingLookups = new AtomicInteger();
        AtomicInteger fa4Lookups = new AtomicInteger();
        ModelConfig config = classicConfig();
        FlashAttentionMetalRoutingPolicy policy = policy(true, countedNullBinding(bindingLookups));

        assertTrue(policy.canUseAttention(config, FlashAttentionModelPolicy.resolve(null, config),
                0, 4, 0, false, countedFa4Availability(fa4Lookups, false)));
        assertEquals(0, bindingLookups.get());
        assertEquals(0, fa4Lookups.get());
    }

    @Test
    void sharedKvLayerRejectsBeforeRuntimeBindingLookup() throws Exception {
        AtomicInteger bindingLookups = new AtomicInteger();
        ModelConfig config = sharedKvConfig();
        FlashAttentionMetalRoutingPolicy policy = policy(true, countedNullBinding(bindingLookups));

        assertFalse(policy.canUseAttention(config, FlashAttentionModelPolicy.resolve(null, config),
                1, 1, 0, false, forbiddenFa4Availability()));
        assertEquals(0, bindingLookups.get());
    }

    @Test
    void slidingLayerRequiresWindowedRuntimeBinding() throws Exception {
        AtomicInteger bindingLookups = new AtomicInteger();
        ModelConfig config = slidingConfig();
        FlashAttentionMetalRoutingPolicy policy = policy(true, countedNullBinding(bindingLookups));

        assertFalse(policy.canUseAttention(config, FlashAttentionModelPolicy.resolve(null, config),
                0, 1, 0, true, forbiddenFa4Availability()));
        assertEquals(1, bindingLookups.get());
    }

    @Test
    void gemma4WithoutLegacyBridgeDelegatesToSlidingPrefillFa4Gate() throws Exception {
        ModelConfig config = slidingGemma4Config();
        FlashAttentionMetalRoutingPolicy policy = policy(true, () -> null);
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, config);

        assertTrue(policy.canUseAttention(config, modelPolicy, 0, 4, 0, false, () -> true));
        assertFalse(policy.canUseAttention(config, modelPolicy, 0, 4, 0, false, () -> false));
    }

    @Test
    void restrictedAttentionTraitsDelegateToSlidingPrefillFa4GateWithoutGemmaModelType() throws Exception {
        ModelConfig config = slidingConfig();
        FlashAttentionMetalRoutingPolicy policy = policy(true, () -> null);
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, config,
                ModelRuntimeTraits.builder()
                        .attention(ModelAttentionTraitsPolicy.nativeBf16Matvec())
                        .build());

        assertFalse(modelPolicy.nativeBf16Matvec());
        assertTrue(policy.canUseAttention(config, modelPolicy, 0, 4, 0, false, () -> true));
        assertFalse(policy.canUseAttention(config, modelPolicy, 0, 4, 0, false, () -> false));
    }

    private static FlashAttentionMetalRoutingPolicy policy(boolean canUseMetal, Supplier<MetalBinding> binding) {
        FlashAttentionRestrictedMetalRoutingPolicy restrictedRouting = new FlashAttentionRestrictedMetalRoutingPolicy(
                binding);
        return new FlashAttentionMetalRoutingPolicy(() -> canUseMetal, binding, restrictedRouting);
    }

    private static Supplier<MetalBinding> countedNullBinding(AtomicInteger lookups) {
        return () -> {
            lookups.incrementAndGet();
            return null;
        };
    }

    private static BooleanSupplier countedFa4Availability(AtomicInteger lookups, boolean available) {
        return () -> {
            lookups.incrementAndGet();
            return available;
        };
    }

    private static BooleanSupplier forbiddenFa4Availability() {
        return () -> {
            throw new AssertionError("FA4 availability should not be checked for this route");
        };
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

    private static ModelConfig sharedKvConfig() throws Exception {
        return new ObjectMapper().readValue("""
                {
                  "model_type": "phi3",
                  "hidden_size": 256,
                  "intermediate_size": 512,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2,
                  "num_kv_shared_layers": 1
                }
                """, ModelConfig.class);
    }

    private static ModelConfig slidingConfig() throws Exception {
        return new ObjectMapper().readValue("""
                {
                  "model_type": "phi3",
                  "hidden_size": 256,
                  "intermediate_size": 512,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2,
                  "sliding_window": 512,
                  "layer_types": ["sliding_attention", "full_attention"]
                }
                """, ModelConfig.class);
    }

    private static ModelConfig slidingGemma4Config() throws Exception {
        return new ObjectMapper().readValue("""
                {
                  "model_type": "gemma4_text",
                  "hidden_size": 256,
                  "intermediate_size": 512,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2,
                  "sliding_window": 512,
                  "layer_types": ["sliding_attention", "full_attention"]
                }
                """, ModelConfig.class);
    }
}
