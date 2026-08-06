/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

final class FlashAttentionDispatchStage {
    private final FlashAttentionDispatchRouter router;
    private final FlashAttentionDispatchExecutor executor;

    FlashAttentionDispatchStage(FlashAttentionDispatchRouter router, FlashAttentionDispatchExecutor executor) {
        this.router = router;
        this.executor = executor;
    }

    AccelTensor compute(FlashAttentionDispatchRequest request) {
        return executor.execute(router.select(request), request);
    }

}
