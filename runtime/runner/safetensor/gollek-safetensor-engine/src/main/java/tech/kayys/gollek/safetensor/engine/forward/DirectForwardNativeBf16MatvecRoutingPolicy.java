/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

record DirectForwardNativeBf16MatvecRoutingPolicy(DirectForwardNativeBf16MatvecOptions options) {

    DirectForwardNativeBf16MatvecRoutingPolicy {
        options = Objects.requireNonNull(options, "options");
    }

    static DirectForwardNativeBf16MatvecRoutingPolicy from(DirectForwardNativeBf16MatvecOptions options) {
        return new DirectForwardNativeBf16MatvecRoutingPolicy(options);
    }

    String describeNativeBf16MatvecPath(int inputDim, int outputDim) {
        if (shouldUseNativeBf16MatvecX8(inputDim, outputDim)) {
            return shouldUseNativeSimdgroupReduction()
                    ? "bf16_matvec_x8_simd"
                    : "bf16_matvec_x8";
        }
        if (shouldUseNativeBf16MatvecX4(inputDim, outputDim)) {
            return shouldUseNativeSimdgroupReduction()
                    ? "bf16_matvec_x4_simd"
                    : "bf16_matvec_x4";
        }
        return "bf16_matvec";
    }

    String describeNativeBf16PairMatvecPath(int inputDim, int outputDim) {
        boolean pairX8 = !options.disableBf16PairX8()
                && shouldUseNativeBf16MatvecX8(inputDim, outputDim);
        if (pairX8) {
            return shouldUseNativeSimdgroupReduction()
                    ? "bf16_pair_matvec_x8_simd"
                    : "bf16_pair_matvec_x8";
        }
        boolean pairX4 = !options.disableBf16PairX4()
                && (options.enableBf16PairX4() || shouldUseNativeBf16MatvecX4(inputDim, outputDim));
        if (pairX4) {
            return shouldUseNativeSimdgroupReduction()
                    ? "bf16_pair_matvec_x4_simd"
                    : "bf16_pair_matvec_x4";
        }
        if (shouldUseNativeBf16PairSimdReduction(inputDim, outputDim)) {
            return "bf16_pair_matvec_simd";
        }
        return "bf16_pair_matvec";
    }

    private boolean shouldUseNativeBf16MatvecX8(int inputDim, int outputDim) {
        if (options.disableBf16MatvecX8() || options.metalMatvecThreads128()) {
            return false;
        }
        return (options.bf16MatvecX8MinInner() <= 0 || inputDim >= options.bf16MatvecX8MinInner())
                && (options.bf16MatvecX8MinOutput() <= 0 || outputDim >= options.bf16MatvecX8MinOutput())
                && (options.bf16MatvecX8MaxOutput() <= 0 || outputDim <= options.bf16MatvecX8MaxOutput());
    }

    private boolean shouldUseNativeBf16MatvecX4(int inputDim, int outputDim) {
        if (options.disableBf16MatvecX4() || options.metalMatvecThreads128()) {
            return false;
        }
        return (options.bf16MatvecX4MinInner() <= 0 || inputDim >= options.bf16MatvecX4MinInner())
                && (options.bf16MatvecX4MinOutput() <= 0 || outputDim >= options.bf16MatvecX4MinOutput())
                && (options.bf16MatvecX4MaxOutput() <= 0 || outputDim <= options.bf16MatvecX4MaxOutput());
    }

    private boolean shouldUseNativeBf16PairSimdReduction(int inputDim, int outputDim) {
        if (options.disableBf16PairSimd()) {
            return false;
        }
        if (options.enableBf16PairSimd()) {
            return true;
        }
        if (options.metalMatvecThreads128()) {
            return false;
        }
        return (options.bf16PairSimdMinInner() <= 0 || inputDim >= options.bf16PairSimdMinInner())
                && (options.bf16PairSimdMinOutput() <= 0 || outputDim >= options.bf16PairSimdMinOutput())
                && (options.bf16PairSimdMaxOutput() <= 0 || outputDim <= options.bf16PairSimdMaxOutput());
    }

    private boolean shouldUseNativeSimdgroupReduction() {
        return !options.disableSimdgroupReduction() && options.enableSimdgroupReduction();
    }
}
