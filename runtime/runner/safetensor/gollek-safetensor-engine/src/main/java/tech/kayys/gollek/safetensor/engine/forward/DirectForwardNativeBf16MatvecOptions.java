/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.envInt;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.envTruthy;

record DirectForwardNativeBf16MatvecOptions(
        boolean metalMatvecThreads128,
        boolean disableBf16MatvecX8,
        boolean disableBf16MatvecX4,
        boolean disableBf16PairX8,
        boolean disableBf16PairX4,
        boolean enableBf16PairX4,
        boolean disableBf16PairSimd,
        boolean enableBf16PairSimd,
        boolean disableSimdgroupReduction,
        boolean enableSimdgroupReduction,
        int bf16MatvecX8MinInner,
        int bf16MatvecX8MinOutput,
        int bf16MatvecX8MaxOutput,
        int bf16MatvecX4MinInner,
        int bf16MatvecX4MinOutput,
        int bf16MatvecX4MaxOutput,
        int bf16PairSimdMinInner,
        int bf16PairSimdMinOutput,
        int bf16PairSimdMaxOutput) {

    static DirectForwardNativeBf16MatvecOptions fromEnvironment() {
        return new DirectForwardNativeBf16MatvecOptions(
                "128".equals(System.getenv("GOLLEK_METAL_MATVEC_THREADS")),
                envTruthy("GOLLEK_METAL_DISABLE_BF16_MATVEC_X8"),
                envTruthy("GOLLEK_METAL_DISABLE_BF16_MATVEC_X4"),
                envTruthy("GOLLEK_METAL_DISABLE_BF16_PAIR_X8"),
                envTruthy("GOLLEK_METAL_DISABLE_BF16_PAIR_X4"),
                envTruthy("GOLLEK_METAL_ENABLE_BF16_PAIR_X4"),
                envTruthy("GOLLEK_METAL_DISABLE_BF16_PAIR_SIMD"),
                envTruthy("GOLLEK_METAL_ENABLE_BF16_PAIR_SIMD"),
                envTruthy("GOLLEK_METAL_DISABLE_SIMDGROUP_REDUCTION"),
                envTruthy("GOLLEK_METAL_ENABLE_SIMDGROUP_REDUCTION"),
                envInt("GOLLEK_METAL_BF16_MATVEC_X8_MIN_INNER", 512),
                envInt("GOLLEK_METAL_BF16_MATVEC_X8_MIN_OUTPUT", 1024),
                envInt("GOLLEK_METAL_BF16_MATVEC_X8_MAX_OUTPUT", 65536),
                envInt("GOLLEK_METAL_BF16_MATVEC_X4_MIN_INNER", 512),
                envInt("GOLLEK_METAL_BF16_MATVEC_X4_MIN_OUTPUT", 1024),
                envInt("GOLLEK_METAL_BF16_MATVEC_X4_MAX_OUTPUT", 8192),
                envInt("GOLLEK_METAL_BF16_PAIR_SIMD_MIN_INNER", 1024),
                envInt("GOLLEK_METAL_BF16_PAIR_SIMD_MIN_OUTPUT", 4096),
                envInt("GOLLEK_METAL_BF16_PAIR_SIMD_MAX_OUTPUT", 0));
    }

    static DirectForwardNativeBf16MatvecOptions defaults() {
        return new DirectForwardNativeBf16MatvecOptions(
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                512,
                1024,
                65536,
                512,
                1024,
                8192,
                1024,
                4096,
                0);
    }

    DirectForwardNativeBf16MatvecOptions withThreads128(boolean enabled) {
        return new DirectForwardNativeBf16MatvecOptions(
                enabled,
                disableBf16MatvecX8,
                disableBf16MatvecX4,
                disableBf16PairX8,
                disableBf16PairX4,
                enableBf16PairX4,
                disableBf16PairSimd,
                enableBf16PairSimd,
                disableSimdgroupReduction,
                enableSimdgroupReduction,
                bf16MatvecX8MinInner,
                bf16MatvecX8MinOutput,
                bf16MatvecX8MaxOutput,
                bf16MatvecX4MinInner,
                bf16MatvecX4MinOutput,
                bf16MatvecX4MaxOutput,
                bf16PairSimdMinInner,
                bf16PairSimdMinOutput,
                bf16PairSimdMaxOutput);
    }

    DirectForwardNativeBf16MatvecOptions withMatvecX8(boolean disabled, int minInner, int minOutput,
            int maxOutput) {
        return new DirectForwardNativeBf16MatvecOptions(
                metalMatvecThreads128,
                disabled,
                disableBf16MatvecX4,
                disableBf16PairX8,
                disableBf16PairX4,
                enableBf16PairX4,
                disableBf16PairSimd,
                enableBf16PairSimd,
                disableSimdgroupReduction,
                enableSimdgroupReduction,
                minInner,
                minOutput,
                maxOutput,
                bf16MatvecX4MinInner,
                bf16MatvecX4MinOutput,
                bf16MatvecX4MaxOutput,
                bf16PairSimdMinInner,
                bf16PairSimdMinOutput,
                bf16PairSimdMaxOutput);
    }

    DirectForwardNativeBf16MatvecOptions withMatvecX4(boolean disabled, int minInner, int minOutput,
            int maxOutput) {
        return new DirectForwardNativeBf16MatvecOptions(
                metalMatvecThreads128,
                disableBf16MatvecX8,
                disabled,
                disableBf16PairX8,
                disableBf16PairX4,
                enableBf16PairX4,
                disableBf16PairSimd,
                enableBf16PairSimd,
                disableSimdgroupReduction,
                enableSimdgroupReduction,
                bf16MatvecX8MinInner,
                bf16MatvecX8MinOutput,
                bf16MatvecX8MaxOutput,
                minInner,
                minOutput,
                maxOutput,
                bf16PairSimdMinInner,
                bf16PairSimdMinOutput,
                bf16PairSimdMaxOutput);
    }

    DirectForwardNativeBf16MatvecOptions withPairRouting(boolean disableX8, boolean disableX4,
            boolean enableX4) {
        return new DirectForwardNativeBf16MatvecOptions(
                metalMatvecThreads128,
                disableBf16MatvecX8,
                disableBf16MatvecX4,
                disableX8,
                disableX4,
                enableX4,
                disableBf16PairSimd,
                enableBf16PairSimd,
                disableSimdgroupReduction,
                enableSimdgroupReduction,
                bf16MatvecX8MinInner,
                bf16MatvecX8MinOutput,
                bf16MatvecX8MaxOutput,
                bf16MatvecX4MinInner,
                bf16MatvecX4MinOutput,
                bf16MatvecX4MaxOutput,
                bf16PairSimdMinInner,
                bf16PairSimdMinOutput,
                bf16PairSimdMaxOutput);
    }

    DirectForwardNativeBf16MatvecOptions withPairSimd(boolean disabled, boolean enabled, int minInner,
            int minOutput, int maxOutput) {
        return new DirectForwardNativeBf16MatvecOptions(
                metalMatvecThreads128,
                disableBf16MatvecX8,
                disableBf16MatvecX4,
                disableBf16PairX8,
                disableBf16PairX4,
                enableBf16PairX4,
                disabled,
                enabled,
                disableSimdgroupReduction,
                enableSimdgroupReduction,
                bf16MatvecX8MinInner,
                bf16MatvecX8MinOutput,
                bf16MatvecX8MaxOutput,
                bf16MatvecX4MinInner,
                bf16MatvecX4MinOutput,
                bf16MatvecX4MaxOutput,
                minInner,
                minOutput,
                maxOutput);
    }

    DirectForwardNativeBf16MatvecOptions withSimdgroupReduction(boolean disabled, boolean enabled) {
        return new DirectForwardNativeBf16MatvecOptions(
                metalMatvecThreads128,
                disableBf16MatvecX8,
                disableBf16MatvecX4,
                disableBf16PairX8,
                disableBf16PairX4,
                enableBf16PairX4,
                disableBf16PairSimd,
                enableBf16PairSimd,
                disabled,
                enabled,
                bf16MatvecX8MinInner,
                bf16MatvecX8MinOutput,
                bf16MatvecX8MaxOutput,
                bf16MatvecX4MinInner,
                bf16MatvecX4MinOutput,
                bf16MatvecX4MaxOutput,
                bf16PairSimdMinInner,
                bf16PairSimdMinOutput,
                bf16PairSimdMaxOutput);
    }
}
