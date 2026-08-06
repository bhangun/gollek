/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

final class DirectForwardNativeBf16MatvecPolicy {
    private static final DirectForwardNativeBf16MatvecOptions OPTIONS =
            DirectForwardNativeBf16MatvecOptions.fromEnvironment();
    private static final DirectForwardNativeBf16MatvecRoutingPolicy ROUTING =
            DirectForwardNativeBf16MatvecRoutingPolicy.from(OPTIONS);

    private DirectForwardNativeBf16MatvecPolicy() {
    }

    static String describeNativeBf16MatvecPath(int inputDim, int outputDim) {
        return ROUTING.describeNativeBf16MatvecPath(inputDim, outputDim);
    }

    static String describeNativeBf16PairMatvecPath(int inputDim, int outputDim) {
        return ROUTING.describeNativeBf16PairMatvecPath(inputDim, outputDim);
    }
}
