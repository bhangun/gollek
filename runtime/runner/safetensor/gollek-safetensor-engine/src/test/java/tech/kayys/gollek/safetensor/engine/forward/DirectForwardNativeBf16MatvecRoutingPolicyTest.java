/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DirectForwardNativeBf16MatvecRoutingPolicyTest {

    @Test
    void defaultsPreferX8WithinThresholds() {
        DirectForwardNativeBf16MatvecRoutingPolicy policy =
                policy(DirectForwardNativeBf16MatvecOptions.defaults());

        assertEquals("bf16_matvec_x8", policy.describeNativeBf16MatvecPath(512, 1024));
        assertEquals("bf16_matvec", policy.describeNativeBf16MatvecPath(511, 1024));
        assertEquals("bf16_matvec", policy.describeNativeBf16MatvecPath(512, 65537));
    }

    @Test
    void x4IsUsedWhenX8DisabledAndThresholdsMatch() {
        DirectForwardNativeBf16MatvecRoutingPolicy policy = policy(
                DirectForwardNativeBf16MatvecOptions.defaults().withMatvecX8(true, 512, 1024, 65536));

        assertEquals("bf16_matvec_x4", policy.describeNativeBf16MatvecPath(512, 1024));
    }

    @Test
    void threads128DisablesX8X4AndPairSimdDefaults() {
        DirectForwardNativeBf16MatvecRoutingPolicy policy = policy(
                DirectForwardNativeBf16MatvecOptions.defaults().withThreads128(true));

        assertEquals("bf16_matvec", policy.describeNativeBf16MatvecPath(4096, 4096));
        assertEquals("bf16_pair_matvec", policy.describeNativeBf16PairMatvecPath(4096, 4096));
    }

    @Test
    void simdgroupSuffixRequiresEnableAndNotDisable() {
        DirectForwardNativeBf16MatvecRoutingPolicy enabled = policy(
                DirectForwardNativeBf16MatvecOptions.defaults().withSimdgroupReduction(false, true));
        DirectForwardNativeBf16MatvecRoutingPolicy disabled = policy(
                DirectForwardNativeBf16MatvecOptions.defaults().withSimdgroupReduction(true, true));

        assertEquals("bf16_matvec_x8_simd", enabled.describeNativeBf16MatvecPath(512, 1024));
        assertEquals("bf16_matvec_x8", disabled.describeNativeBf16MatvecPath(512, 1024));
    }

    @Test
    void pairDefaultsPreferX8ThenX4WhenForced() {
        DirectForwardNativeBf16MatvecRoutingPolicy x4Forced = policy(
                DirectForwardNativeBf16MatvecOptions.defaults().withPairRouting(true, false, true));

        assertEquals("bf16_pair_matvec_x8",
                policy(DirectForwardNativeBf16MatvecOptions.defaults()).describeNativeBf16PairMatvecPath(512, 1024));
        assertEquals("bf16_pair_matvec_x4", x4Forced.describeNativeBf16PairMatvecPath(1, 1));
    }

    @Test
    void pairSimdFallbackCanBeThresholdOrExplicit() {
        DirectForwardNativeBf16MatvecOptions thresholdMatched =
                DirectForwardNativeBf16MatvecOptions.defaults()
                        .withPairRouting(true, true, false)
                        .withPairSimd(false, false, 1024, 4096, 0);
        DirectForwardNativeBf16MatvecRoutingPolicy forced =
                policy(thresholdMatched.withPairSimd(false, true, 1024, 4096, 0));
        DirectForwardNativeBf16MatvecRoutingPolicy disabled =
                policy(thresholdMatched.withPairSimd(true, true, 1024, 4096, 0));

        assertEquals("bf16_pair_matvec_simd",
                policy(thresholdMatched).describeNativeBf16PairMatvecPath(1024, 4096));
        assertEquals("bf16_pair_matvec_simd", forced.describeNativeBf16PairMatvecPath(1, 1));
        assertEquals("bf16_pair_matvec", disabled.describeNativeBf16PairMatvecPath(1024, 4096));
    }

    private static DirectForwardNativeBf16MatvecRoutingPolicy policy(
            DirectForwardNativeBf16MatvecOptions options) {
        return DirectForwardNativeBf16MatvecRoutingPolicy.from(options);
    }
}
