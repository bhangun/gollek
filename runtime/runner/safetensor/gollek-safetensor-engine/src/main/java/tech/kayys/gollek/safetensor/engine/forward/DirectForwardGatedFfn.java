/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

final class DirectForwardGatedFfn {
    private DirectForwardGatedFfn() {
    }

    static AccelTensor forward(DirectForwardGatedFfnRequest request) {
        AccelTensor completeFastPath = DirectForwardGatedFfnFastPaths.tryComplete(request);
        if (completeFastPath != null) {
            return completeFastPath;
        }

        return DirectForwardGatedFfnFallback.forward(request);
    }

}
